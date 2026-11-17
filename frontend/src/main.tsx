import {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

type Progress = {phase?: string; completed?: number; total?: number; step?: number; target?: number;
  epoch?: number; loss?: number; learning_rate?: number; peak_cuda_bytes?: number; eta_seconds?: number};
type Job = {id: string; kind: string; state: string; result?: {[key: string]: any}; error?: string; latest_progress?: Progress};
type ModelRef = {model_id: string; version_id: string};
type Dataset = {role: string; dataset_id: string; version_id: string; stats: Record<string, number>;
  features_ready: boolean; initial_model?: ModelRef};
type Model = ModelRef & {role: string; architecture: string; speakers: string[]; present: boolean; step?: number; source: string};
type Song = {song_id: number; title: string; artists: string | string[]; album: string; aliases: string[]; available: boolean; duration_seconds: number};
type Search = {search_id: string; candidates: Song[]; page: number; quality: string};
type AudioFormat = 'wav' | 'mp3';
type Saved = {artifact_id: string; title: string; model: Partial<ModelRef>; model_role?: string; song_title?: string;
  stems: boolean; format: AudioFormat};
type SelectJob = (path: string, payload?: object) => Promise<Job | undefined>;
const active = (job?: Job | null) => job && ['QUEUED', 'RUNNING'].includes(job.state);
const modelKey = (model: ModelRef) => model.model_id + '/' + model.version_id;
const states: Record<string, string> = {QUEUED: '排队中', RUNNING: '运行中', COMPLETED: '完成', FAILED: '失败',
  INTERRUPTED: '已中断', CANCELLED: '已取消', PAUSED: '已暂停', EARLY_COMPLETED: '已提前结束'};
const kinds: Record<string, string> = {diagnostic: '系统诊断', dataset_scan: '扫描', dataset_prepare: '整理', features: '提取特征',
  train: '训练', model_import: '导入模型', model_check: '开启模型', test: '干声试听', cover: '生成翻唱',
  remix: '重混音', export: '发布模型', asset_base: '底模准备', dataset_append: '追加人声到数据集'};
const label = (state: string) => states[state] ?? state;
const jobLabel = (kind: string) => kinds[kind] ?? kind;
const phases: Record<string, string> = {PREPARING: '准备训练', TRAINING: '训练', VALIDATING: '验证', CHECKPOINT: '保存恢复点',
  FEATURES: '提取特征', DOWNLOADING: '下载音源', DECODING: '解码歌曲', SEPARATING: '分离人声',
  PREPARING_VOCALS: '准备人声', LOADING_RIFT: '加载 RIFT', CONVERTING: '转换人声', MIXING: '混音',
  MODEL_PUBLISHED: '模型已发布', EXPORT_FAILED: '导出失败', CLIPPING: '裁剪人声'};

function TaskProgress({progress}: {progress?: Progress}) {
  if (!progress) return null;
  const step = progress.step ?? progress.completed;
  const total = progress.target ?? progress.total;
  return <div>
    {progress.phase && <p>{phases[progress.phase] ?? progress.phase}
      {progress.epoch === undefined ? '' : ' · epoch ' + (progress.epoch + 1)}
      {progress.loss == null ? '' : ' · loss ' + progress.loss.toPrecision(4)}
      {progress.learning_rate === undefined ? '' : ' · LR ' + progress.learning_rate.toExponential(2)}
      {progress.peak_cuda_bytes === undefined ? '' : ' · CUDA 峰值 ' + (progress.peak_cuda_bytes / 2 ** 30).toFixed(2) + ' GiB'}
      {progress.eta_seconds === undefined ? '' : ' · 预计剩余 ' + Math.ceil(progress.eta_seconds) + ' 秒'}
    </p>}
    {step !== undefined && total !== undefined && total > 0 &&
      <div className="actions"><progress max={total} value={step} aria-label="已完成进度"/><span>{step} / {total}</span></div>}
  </div>;
}

async function api<T>(path: string, payload?: unknown, method?: string): Promise<T> {
  const response = await fetch('/api' + path, payload === undefined && method === undefined ? undefined : {
    method: method ?? 'POST', headers: {'Content-Type': 'application/json'},
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail));
  return data;
}
const requestId = () => ({request_id: crypto.randomUUID()});

type FileList = {path: string; parent: string; entries: {path: string; name: string; directory: boolean}[]};
function PathField({name, value, onChange, kind}: {
  name: string; value: string; onChange: (value: string) => void; kind: 'directory' | 'model' | 'audio';
}) {
  const [files, setFiles] = useState<FileList>();
  const [error, setError] = useState('');
  const [picking, setPicking] = useState(false);
  const browse = async (path: string) => {
    setError('');
    if (kind === 'directory') {
      setPicking(true);
      try {
        const result = await api<{path: string | null}>('/paths/pick-directory', {...requestId(), source: path || null});
        if (result.path) onChange(result.path);
      } catch (e) {setError((e as Error).message);}
      finally {setPicking(false);}
      return;
    }
    try {setFiles(await api<FileList>('/paths?kind=' + kind + '&path=' + encodeURIComponent(path)));}
    catch (e) {setError((e as Error).message);}
  };
  const select = (path: string) => {onChange(path); setFiles(undefined);};
  return <>
    <label>{name}<input value={value} onChange={e => onChange(e.target.value)}/></label>
    <button disabled={picking} onClick={() => files ? setFiles(undefined) : browse(value)}>{picking ? '等待选择目录…' : '浏览服务电脑'}</button>
    {error && <p role="alert">{error}</p>}
    {files && <div className="file-list">
      <p>{files.path}</p>
      <div className="actions">
        <button disabled={files.path === files.parent} onClick={() => browse(files.parent)}>上一级</button>
      </div>
      <ul>{files.entries.map(file => <li key={file.path}>
        <button onClick={() => file.directory ? browse(file.path) : select(file.path)}>
          {file.directory ? file.name + '/' : file.name}
        </button>
      </li>)}</ul>
    </div>}
  </>;
}

function ModelChooser({models, model, speaker, setModel, setSpeaker}: {
  models: Model[]; model?: Model; speaker: string; setModel: (value: string) => void; setSpeaker: (value: string) => void;
}) {
  return <div className="grid">
    <label>模型版本<select value={model ? modelKey(model) : ''} onChange={e => {
      const choice = models.find(row => modelKey(row) === e.target.value);
      setModel(e.target.value); setSpeaker(choice?.speakers[0] ?? '');
    }}><option value="">选择模型</option>{models.map(row =>
      <option key={modelKey(row)} value={modelKey(row)} disabled={!row.present}>
        {row.role} · {row.architecture}{row.step === undefined ? '' : ' · step ' + row.step} · {row.version_id.slice(0, 8)}
      </option>)}</select></label>
    <label>说话人<select value={speaker} onChange={e => setSpeaker(e.target.value)} disabled={!model}>
      {model?.speakers.map(value => <option key={value}>{value}</option>)}
    </select></label>
  </div>;
}

function Training({datasets, submit, selected, requestedVersion}: {
  datasets: Dataset[]; submit: SelectJob; selected?: Job; requestedVersion: string;
}) {
  const [source, setSource] = useState('');
  const [role, setRole] = useState('');
  const [language, setLanguage] = useState('zh');
  const [recursive, setRecursive] = useState(true);
  const [excluded, setExcluded] = useState<string[]>([]);
  const [version, setVersion] = useState('');
  const [useBase, setUseBase] = useState(false);
  const [steps, setSteps] = useState(30000);
  const [mode, setMode] = useState<'steps' | 'epochs'>('steps');
  const [epochs, setEpochs] = useState(1);
  const [batch, setBatch] = useState(16);
  const [accumulation, setAccumulation] = useState(1);
  const [lr, setLR] = useState(0.00002);
  const [workers, setWorkers] = useState(4);
  const [preflight, setPreflight] = useState<object>();
  const [checkError, setCheckError] = useState('');
  useEffect(() => {setVersion(requestedVersion); setUseBase(false);}, [requestedVersion]);
  const data = datasets.find(row => row.version_id === version);
  const spec = {source, role, language, recursive, excluded_paths: excluded};
  const cfg = data && {dataset_id: data.dataset_id, version_id: version, stop_mode: mode,
    max_train_steps: mode === 'steps' ? steps : null, max_train_epochs: mode === 'epochs' ? epochs : null,
    train_batch_size: batch, gradient_accumulation_steps: accumulation, learning_rate: lr, num_workers: workers,
    use_base_model: useBase};
  const report = ['dataset_scan', 'dataset_prepare'].includes(selected?.kind ?? '') ? selected?.result : undefined;
  return <>
    <section>
      <h2>录音数据</h2>
      <PathField name="服务电脑上的录音目录" value={source} onChange={value => {setSource(value); setExcluded([]);}} kind="directory"/>
      <div className="grid">
        <label>角色<input value={role} onChange={e => setRole(e.target.value)}/></label>
        <label>语言<input value={language} onChange={e => setLanguage(e.target.value)}/></label>
      </div>
      <label className="check"><input type="checkbox" checked={recursive} onChange={e => setRecursive(e.target.checked)}/>扫描子目录</label>
      <div className="actions">
        <button disabled={!role || !source} onClick={() => submit('/datasets/scan', spec)}>扫描检测</button>
        <button disabled={!role || !source} onClick={() => submit('/datasets/prepare', spec)}>整理数据集</button>
      </div>
      {report && <>
        <p>可解码 {report.stats.decoded_seconds.toFixed(1)} 秒 · 保留 {report.stats.retained_seconds.toFixed(1)} 秒 · 排除 {report.stats.excluded_seconds.toFixed(1)} 秒</p>
        <details><summary>文件报告 · {report.sources.length} 项</summary>
          <ul>{report.sources.map((item: any, index: number) => <li key={item.path}>
            {item.path} · {item.reason ?? '可用'} · 保留 {item.retained_seconds.toFixed(1)} 秒
            {item.seconds !== undefined && <>
              <audio controls preload="none" src={'/api/datasets/previews/' + selected?.id + '/' + index}/>
              <label className="check"><input type="checkbox" checked={excluded.includes(item.path)} onChange={e =>
                setExcluded(list => e.target.checked ? [...list, item.path] : list.filter(path => path !== item.path))
              }/>整理时排除</label>
            </>}
          </li>)}</ul>
        </details>
      </>}
    </section>
    <section>
      <h2>常规微调</h2><p>RIFT-SVC V3 · 1024-16 · BF16 · 验证 batch 1</p>
      <label>数据版本<select value={data?.version_id ?? ''} onChange={e => {setVersion(e.target.value); setUseBase(false); setPreflight(undefined); setCheckError('');}}>
        <option value="">选择整理好的数据集</option>
        {datasets.map(row => <option key={row.version_id} value={row.version_id}>{row.role} · {row.version_id.slice(0, 8)} · 训练 {row.stats.train_clips} 段{row.initial_model ? ' · 已追加人声' : ''}</option>)}
      </select></label>
      {data?.features_ready ? <p>训练特征已就绪</p> :
        <button disabled={!data} onClick={() => data && submit('/features', {dataset_id: data.dataset_id, version_id: version})}>准备训练特征</button>}
      {data?.initial_model && <label>从哪里开始微调<select value={useBase ? 'base' : 'trained'} onChange={e => {
        setUseBase(e.target.value === 'base'); setPreflight(undefined); setCheckError('');
      }}>
        <option value="trained">来源模型 · {data.initial_model.version_id.slice(0, 8)}</option>
        <option value="base">官方底模</option>
      </select></label>}
      <div className="grid">
        <label>物理 batch<input type="number" min="1" max="128" value={batch} onChange={e => setBatch(e.target.valueAsNumber)}/></label>
        <label>梯度累积<input type="number" min="1" max="64" value={accumulation} onChange={e => setAccumulation(e.target.valueAsNumber)}/></label>
        <label>停止条件<select value={mode} onChange={e => setMode(e.target.value as typeof mode)}><option value="steps">累计优化器步数</option><option value="epochs">累计轮次</option></select></label>
        {mode === 'steps' ? <label>目标步数<input type="number" min="1" value={steps} onChange={e => setSteps(e.target.valueAsNumber)}/></label>
          : <label>目标轮次<input type="number" min="1" value={epochs} onChange={e => setEpochs(e.target.valueAsNumber)}/></label>}
      </div>
      <details><summary>高级训练参数</summary><div className="grid">
        <label>学习率<input type="number" min="0.00000001" step="0.00000001" value={lr} onChange={e => setLR(e.target.valueAsNumber)}/></label>
        <label>数据加载 worker<input type="number" min="0" max="8" value={workers} onChange={e => setWorkers(e.target.valueAsNumber)}/></label>
      </div></details>
      <p>有效 batch {batch * accumulation}{data ? ' · 预估每轮训练批次 ' + Math.floor(data.stats.train_clips / batch) : ''}</p>
      <div className="actions">
        <button disabled={!cfg} onClick={() => {
          setCheckError(''); setPreflight(undefined);
          if (cfg) api<object>('/training/preflight', cfg).then(setPreflight).catch(e => setCheckError(e.message));
        }}>检查训练配置与显存</button>
        <button disabled={!cfg || !data?.features_ready || !data.stats.validation_clips || data.stats.train_clips < batch}
          onClick={() => cfg && submit('/training', cfg)}>开始训练</button>
      </div>
      {checkError && <p role="alert">{checkError}</p>}
      {preflight && <pre>{JSON.stringify(preflight, null, 2)}</pre>}
    </section>
  </>;
}

function ManagedTraining({datasets, models, reportError, refresh}: {
  datasets: Dataset[]; models: Model[]; reportError: (value: string) => void; refresh: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const trained = models.filter(model => model.source !== 'external');
  const remove = async (path: string, message: string) => {
    if (!window.confirm(message)) return;
    reportError(''); setBusy(true);
    try {await api(path, undefined, 'DELETE'); refresh();}
    catch (e) {reportError((e as Error).message);}
    finally {setBusy(false);}
  };
  return <section>
    <h2>模型与数据集管理</h2>
    <details><summary>已训练模型 · {trained.length} 个版本</summary>
      <p>删除发布的模型版本；训练恢复点和已保存作品仍可使用。</p>
      {trained.map(row => <div className="manage-row" key={modelKey(row)}>
        <span>{row.role} · {row.version_id.slice(0, 8)} · {row.step ?? '—'} 步</span>
        <button className="danger" disabled={busy} onClick={() => remove('/models/' + modelKey(row),
          `删除「${row.role}」的模型版本 ${row.version_id.slice(0, 8)}？`)}>删除模型</button>
      </div>)}
    </details>
    <details><summary>已整理数据集 · {datasets.length} 个版本</summary>
      <p>删除所选版本的整理产物和特征，保留原始录音目录及其他版本。</p>
      {datasets.map(row => <div className="manage-row" key={row.version_id}>
        <span>{row.role} · {row.version_id.slice(0, 8)} · 训练 {row.stats.train_clips} 段 · 验证 {row.stats.validation_clips} 段</span>
        <button className="danger" disabled={busy} onClick={() => remove('/datasets/' + row.dataset_id + '/' + row.version_id,
          `删除「${row.role}」的整理版本 ${row.version_id.slice(0, 8)}？原始录音会保留。`)}>删除数据集</button>
      </div>)}
    </details>
  </section>;
}

const stemTracks = [{id: 'vocals', label: '转换人声'}, {id: 'instrumental', label: '伴奏'}];
const stamp = (time: number) => {
  const ticks = Math.round(time * 100);
  return Math.floor(ticks / 6000) + ':' + ((ticks % 6000) / 100).toFixed(2).padStart(5, '0');
};

function CoverPlayer({audio, cover, submit, jobs, openTraining}: {
  audio: string; cover: Job; submit: SelectJob; jobs: Job[]; openTraining: (version: string) => void;
}) {
  const player = useRef<HTMLAudioElement>(null);
  const original = useRef<HTMLAudioElement>(null);
  const previewEnd = useRef<number | undefined>(undefined);
  const previewTimer = useRef<number | undefined>(undefined);
  const choosingWithPointer = useRef(false);
  const [duration, setDuration] = useState(0);
  const [originalReady, setOriginalReady] = useState(false);
  const [time, setTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [cut, setCut] = useState<{anchor: number; resume: boolean; point?: number}>();
  const [draft, setDraft] = useState<number>();
  const [fragment, setFragment] = useState('');
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const job = jobs.find(row => row.id === fragment);
  const training = cover.result?.training_dataset;
  const canConfirm = cut?.point !== undefined && Math.abs(cut.point - cut.anchor) >= 0.5;
  useEffect(() => {
    const media = player.current;
    const source = original.current;
    return () => {
      window.clearTimeout(previewTimer.current);
      media?.pause();
      source?.pause();
    };
  }, []);

  const play = () => {
    original.current?.pause();
    return player.current?.play().catch(error => {
      if (error.name !== 'AbortError') setError('无法播放翻唱');
    });
  };
  const clearPreview = () => {
    window.clearTimeout(previewTimer.current);
    previewEnd.current = undefined;
  };
  const stopPreview = (media: HTMLAudioElement) => {
    clearPreview();
    media.pause();
    setTime(media.currentTime);
  };
  const schedulePreviewStop = (media: HTMLAudioElement) => {
    const end = previewEnd.current;
    if (end === undefined || media.paused) return;
    const remaining = (end - media.currentTime) / media.playbackRate;
    if (remaining <= 0) {stopPreview(media); return;}
    window.clearTimeout(previewTimer.current);
    previewTimer.current = window.setTimeout(() => stopPreview(media), remaining * 1000);
  };
  const compareOriginal = () => {
    const source = original.current;
    const media = player.current;
    if (!source || !media || !originalReady) return;
    setError('');
    stopPreview(media);
    source.currentTime = Math.min(media.currentTime, source.duration);
    void source.play().catch(error => {
      if (error.name !== 'AbortError') setError('无法播放原唱');
    });
  };
  const begin = () => {
    const media = player.current;
    if (!media) return;
    setError('');
    original.current?.pause();
    setCut({anchor: media.currentTime, resume: !media.paused && !media.ended});
    media.pause();
  };
  const previewPoint = (point: number) => {
    const media = player.current;
    if (!media || !cut) return;
    clearPreview();
    setError('');
    setCut({...cut, point});
    media.pause();
    media.currentTime = point;
    setTime(point);
    const end = Math.min(point + 10, media.duration);
    if (end > point) {
      previewEnd.current = end;
      play();
    }
  };
  const finishCut = (anchor: number, resume: boolean) => {
    clearPreview();
    choosingWithPointer.current = false;
    setCut(undefined);
    setDraft(undefined);
    setError('');
    const media = player.current;
    if (!media) return;
    media.pause();
    media.currentTime = anchor;
    setTime(anchor);
    if (resume) play();
  };
  const confirmPoint = async () => {
    if (!cut || cut.point === undefined || !canConfirm || sending) return;
    setSending(true);
    finishCut(cut.anchor, true);
    try {
      const result = await submit('/covers/' + cover.id + '/dataset-fragment',
        {first_seconds: cut.anchor, second_seconds: cut.point});
      if (result) setFragment(result.id);
    } finally {setSending(false);}
  };
  return <>
    <audio ref={player} src={audio} preload="metadata"
      onLoadedMetadata={e => setDuration(e.currentTarget.duration)}
      onTimeUpdate={e => {
        const media = e.currentTarget;
        if (previewEnd.current !== undefined && media.currentTime >= previewEnd.current) stopPreview(media);
        else setTime(media.currentTime);
      }}
      onPlaying={e => schedulePreviewStop(e.currentTarget)}
      onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)}
      onEnded={e => {if (e.currentTarget.ended) {clearPreview(); setPlaying(false);}}}/>
    <div className="actions player">
      <button disabled={!duration || Boolean(cut)} onClick={() => playing ? player.current?.pause() : play()}>{playing ? '暂停' : '播放'}</button>
      <input type="range" aria-label={cut ? '选择剪切终点' : '播放进度'} min={0} max={duration || 0} step="any"
        disabled={!duration} value={cut ? draft ?? cut.point ?? cut.anchor : time}
        onPointerDown={e => {if (cut && e.button === 0) choosingWithPointer.current = true;}}
        onChange={e => {
          const point = e.currentTarget.valueAsNumber;
          if (cut && choosingWithPointer.current) setDraft(point);
          else if (cut) previewPoint(point);
          else if (player.current) {player.current.currentTime = point; setTime(point);}
        }}
        onPointerUp={e => {
          if (!cut || e.button !== 0) return;
          choosingWithPointer.current = false;
          setDraft(undefined);
          previewPoint(e.currentTarget.valueAsNumber);
        }}
        onPointerCancel={() => {choosingWithPointer.current = false; setDraft(undefined);}}
        onKeyDown={() => {choosingWithPointer.current = false; setDraft(undefined);}}/>
      <span>{cut?.point !== undefined ? '试听 ' : ''}{stamp(time)} / {stamp(duration)}</span>
    </div>
    {cover.result?.original_available && <div className="actions">
      <strong>原唱</strong>
      <audio ref={original} controls preload="metadata" aria-label="原唱" src={audio + '&track=original'}
        onLoadedMetadata={e => setOriginalReady(Number.isFinite(e.currentTarget.duration))}
        onPlay={e => {if (!e.currentTarget.paused && player.current) stopPreview(player.current);}}/>
      <button disabled={!duration || !originalReady} onClick={compareOriginal}>从翻唱对应位置听原唱</button>
    </div>}
    {cut ? <>
      <p>原播放位置 {stamp(cut.anchor)}。点击进度条可从选点试听最多 10 秒，也可重新选点。确认后，两点之间的人声将加入「{training?.role}」的数据集。</p>
      {cut.point !== undefined && <p>当前选点 {stamp(cut.point)}{!canConfirm && ' · 两点需相隔至少 0.5 秒'}</p>}
      <div className="actions">
        <button disabled={!canConfirm || sending} onClick={confirmPoint}>确定剪切</button>
        <button onClick={() => finishCut(cut.anchor, cut.resume)}>取消剪切</button>
      </div>
    </> : <button disabled={!training || !duration || sending || Boolean(active(job))} onClick={begin}>剪切片段</button>}
    {!training && <p>此结果没有可追加的项目训练数据集。</p>}
    {error && <p role="alert">{error}</p>}
    {job && <p aria-live="polite">{job.result ?
      <>已追加 {job.result.added_clips} 段：{job.result.previous_train_clips} → {job.result.train_clips} 段。
        <button onClick={() => openTraining(job.result!.version_id)}>用新版本微调</button></> :
      <>追加人声：{label(job.state)} {job.error}</>}</p>}
    {cover.result?.tracks?.length > 0 && <details><summary>试听或下载独立音轨</summary>
      {stemTracks.filter(track => cover.result?.tracks.includes(track.id)).map(track => {
        const url = audio + '&track=' + track.id;
        return <div key={track.id} className="actions"><span>{track.label}</span><audio controls preload="none" src={url}/>
          <a href={url + '&download=true'}>下载{track.label}</a></div>;
      })}
    </details>}
  </>;
}

function SavedLibrary({saved, models, reportError, refresh}: {
  saved: Saved[]; models: Model[]; reportError: (value: string) => void; refresh: () => void;
}) {
  const [page, setPage] = useState(1);
  const [modelFilter, setModelFilter] = useState('');
  const [songFilter, setSongFilter] = useState('');
  const [batch, setBatch] = useState(false);
  const [chosen, setChosen] = useState<string[]>([]);
  const [removed, setRemoved] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const removedIds = new Set(removed);
  const chosenIds = new Set(chosen);
  const entries = saved.filter(item => !removedIds.has(item.artifact_id));
  const modelsById = new Map(models.map(row => [row.model_id, row.role]));
  const modelId = (row: Saved) => row.model.model_id || 'unknown';
  const role = (row: Saved) => row.model_role || modelsById.get(modelId(row)) ||
    (row.model.model_id ? '模型 ' + row.model.model_id.slice(0, 8) : '未知模型');
  const song = (row: Saved) => row.song_title || row.title;
  const modelOptions = Array.from(new Map(entries.map(row => [modelId(row), role(row)])));
  const songOptions = Array.from(new Set(entries.map(song))).sort((a, b) => a.localeCompare(b, 'zh'));
  const filtered = entries.filter(row => (!modelFilter || modelId(row) === modelFilter) && (!songFilter || song(row) === songFilter));
  const pages = Math.max(1, Math.ceil(filtered.length / 10));
  const visiblePage = Math.min(page, pages);
  const visible = filtered.slice((visiblePage - 1) * 10, visiblePage * 10);
  const selection = filtered.map(row => row.artifact_id).filter(id => chosenIds.has(id));
  const changeFilter = (set: (value: string) => void, value: string) => {set(value); setPage(1); setChosen([]);};
  const remove = async (ids: string[], single?: string) => {
    if (!window.confirm(single ? `删除作品「${single}」？` : `删除已选的 ${ids.length} 个作品？`)) return;
    reportError(''); setBusy(true);
    try {
      if (single) await api('/saved-covers/' + ids[0], undefined, 'DELETE');
      else await api('/saved-covers/bulk-delete', {ids});
      const deleted = new Set(ids);
      setRemoved(rows => [...rows, ...ids]); setChosen(rows => rows.filter(id => !deleted.has(id))); refresh();
    } catch (e) {reportError((e as Error).message);}
    finally {setBusy(false);}
  };
  return <section>
    <h2>已保存作品</h2>
    <div className="grid">
      <label>按模型筛选<select value={modelFilter} onChange={e => changeFilter(setModelFilter, e.target.value)}>
        <option value="">全部模型</option>
        {modelFilter && !modelOptions.some(([id]) => id === modelFilter) && <option value={modelFilter}>当前筛选（无作品）</option>}
        {modelOptions.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
      </select></label>
      <label>按歌曲筛选<select value={songFilter} onChange={e => changeFilter(setSongFilter, e.target.value)}>
        <option value="">全部歌曲</option>
        {songFilter && !songOptions.includes(songFilter) && <option>{songFilter}</option>}
        {songOptions.map(name => <option key={name}>{name}</option>)}
      </select></label>
    </div>
    <div className="actions">
      <button disabled={busy || (!entries.length && !batch)} onClick={() => {setBatch(!batch); setChosen([]);}}>{batch ? '退出批量删除' : '批量删除'}</button>
      {batch && <>
        <button disabled={busy || !visible.length} onClick={() => setChosen(rows => Array.from(new Set([...rows, ...visible.map(row => row.artifact_id)])))}>全选本页</button>
        <button disabled={busy || !filtered.length} onClick={() => setChosen(filtered.map(row => row.artifact_id))}>全选筛选结果</button>
        <button disabled={busy || !selection.length} onClick={() => setChosen([])}>取消全选</button>
        <button className="danger" disabled={busy || !selection.length} onClick={() => remove(selection)}>删除选中（{selection.length}）</button>
      </>}
    </div>
    {visible.map(item => {
      const url = '/api/saved-covers/' + item.artifact_id + '/audio';
      return <div className="saved-row" key={item.artifact_id}>
        <div className="manage-row">
          {batch ? <label className="check"><input type="checkbox" checked={chosenIds.has(item.artifact_id)} disabled={busy}
            onChange={e => setChosen(rows => e.target.checked ? [...rows, item.artifact_id] : rows.filter(id => id !== item.artifact_id))}/>{item.title}</label>
            : <strong>{item.title}</strong>}
          <span>{role(item)} · {song(item)} · {item.format.toUpperCase()}</span>
          {!batch && <button className="danger" disabled={busy} onClick={() => remove([item.artifact_id], item.title)}>删除作品</button>}
        </div>
        <audio controls preload="none" src={url}/>
        <div className="actions">
          <a href={url + '?download=true'}>下载 {item.format.toUpperCase()}</a>
          {item.stems && stemTracks.map(track =>
            <a key={track.id} href={url + '?download=true&track=' + track.id}>下载{track.label}</a>)}
        </div>
      </div>;
    })}
    {!visible.length && <p>没有符合条件的作品。</p>}
    <div className="actions">
      <button disabled={visiblePage <= 1} onClick={() => setPage(visiblePage - 1)}>上一页</button>
      <span>第 {visiblePage} / {pages} 页 · 共 {filtered.length} 个作品</span>
      <button disabled={visiblePage >= pages} onClick={() => setPage(visiblePage + 1)}>下一页</button>
    </div>
  </section>;
}

function Covers({models, model, speaker, choose, selectSpeaker, submit, activeModel, cover, saved, jobs, openTraining, reportError, refresh}: {
  models: Model[]; model?: Model; speaker: string; choose: (value: string) => void; selectSpeaker: (value: string) => void;
  submit: SelectJob; activeModel: {job?: Job; model?: Model}; cover: Job | null; saved: Saved[]; jobs: Job[];
  openTraining: (version: string) => void;
  reportError: (value: string) => void; refresh: () => void;
}) {
  const [external, setExternal] = useState('');
  const [local, setLocal] = useState('');
  const [query, setQuery] = useState('');
  const [search, setSearch] = useState<Search>();
  const [searching, setSearching] = useState(false);
  const [song, setSong] = useState<Song>();
  const [steps, setSteps] = useState(64);
  const [transpose, setTranspose] = useState(0);
  const [octave, setOctave] = useState(0);
  const [vocal, setVocal] = useState(0);
  const [backing, setBacking] = useState(0);
  const [title, setTitle] = useState('');
  const [stems, setStems] = useState(true);
  const [format, setFormat] = useState<AudioFormat>('wav');
  const [saving, setSaving] = useState(false);
  const [pending, setPending] = useState(false);
  const [account, setAccount] = useState('');
  const variants = search?.candidates ?? [];
  const audio = cover?.state === 'COMPLETED' ? '/api/covers/' + cover.id + '/audio' : undefined;
  const request = {inference: {inference_steps: steps, global_transpose_semitones: transpose, vocal_octave_offset: octave},
    instrumental_gain_db: backing, vocal_gain_db: vocal};
  const find = async (page: number) => {
    setSearching(true); setSong(undefined);
    try {
      setSearch(await api<Search>('/music/search?query=' + encodeURIComponent(query.trim()) + '&page=' + page));
    } catch (e) {reportError((e as Error).message);}
    finally {setSearching(false);}
  };
  const generate = async (source: object) => {
    setPending(true);
    try {await submit('/covers', {...request, ...source});}
    finally {setPending(false);}
  };
  const save = async () => {
    if (!cover || !title.trim()) return;
    setSaving(true);
    try {await api('/covers/' + cover.id + '/save', {...requestId(), title: title.trim(), stems, format}); refresh();}
    catch (e) {reportError((e as Error).message);}
    finally {setSaving(false);}
  };
  return <>
    <section>
      <h2>翻唱模型</h2>
      <ModelChooser models={models} model={model} speaker={speaker} setModel={choose} setSpeaker={selectSpeaker}/>
      <div className="actions">
        <button disabled={!model || !speaker} onClick={() => model && submit('/models/enabled', {model_id: model.model_id, version_id: model.version_id, speaker})}>开启所选模型</button>
        {activeModel.job && <button onClick={() => api('/models/enabled', undefined, 'DELETE').then(refresh).catch(e => reportError(e.message))}>关闭翻唱</button>}
      </div>
      <p>{activeModel.model ? '翻唱已开启 · ' + activeModel.model.architecture : activeModel.job ? label(activeModel.job.state) + ' ' + (activeModel.job.error ?? '') : '请开启模型后生成翻唱'}</p>
      <details><summary>导入外部 RIFT 模型</summary>
        <PathField name="服务电脑上的 .ckpt 路径" value={external} onChange={setExternal} kind="model"/>
        <button disabled={!external} onClick={() => submit('/models/import', {path: external})}>验证并导入</button>
      </details>
      <div className="grid">
        <label>推理采样步数<input type="number" min="1" max="256" value={steps} onChange={e => setSteps(e.target.valueAsNumber)}/></label>
        <label>整曲移调（半音）<input type="number" min="-12" max="12" value={transpose} onChange={e => setTranspose(e.target.valueAsNumber)}/></label>
        <label>人声音区<select value={octave} onChange={e => setOctave(Number(e.target.value))}><option value="-12">低一八度</option><option value="0">原音区</option><option value="12">高一八度</option></select></label>
      </div>
    </section>
    <section>
      <h2>选择歌曲版本</h2>
      <div className="actions"><button onClick={() => api<{configured: boolean; online_verified: boolean}>('/music/status?verify=true')
        .then(value => setAccount(value.online_verified ? '网易云会话有效' : '需要网易云会话')).catch(e => reportError(e.message))}>检查网易云会话</button>{account && <span>{account}</span>}</div>
      <label>歌名<input value={query} onChange={e => setQuery(e.target.value)} onKeyDown={e => {if (e.key === 'Enter' && query.trim() && !searching) find(1);}}/></label>
      <button disabled={!query.trim() || searching} onClick={() => find(1)}>{searching ? '搜索中…' : '搜索网易云'}</button>
      {search && <div className="songs">
        <p>音源 {search.quality}</p>
        {variants.length === 0 && <p>本页无匹配结果</p>}
        {variants.map(item => <label className="song" key={item.song_id}>
          <input type="radio" name="song" disabled={!item.available} checked={song?.song_id === item.song_id} onChange={() => setSong(item)}/>
          <span><strong>{item.title}</strong> · {Array.isArray(item.artists) ? item.artists.join(' / ') : item.artists}
            <small>{item.album} {item.aliases.join(' / ')} · {Math.floor(item.duration_seconds / 60)}:{Math.floor(item.duration_seconds % 60).toString().padStart(2, '0')} · {item.available ? '完整音源' : '完整音源不可用'}</small></span>
        </label>)}
        <div className="actions">
          <button disabled={search.page <= 1 || searching} onClick={() => find(search.page - 1)}>上一页</button>
          <button disabled={search.page >= 5 || variants.length === 0 || searching} onClick={() => find(search.page + 1)}>下一页</button>
          <button disabled={!song || !activeModel.model || Boolean(active(cover)) || pending}
            onClick={() => song && generate({search_id: search.search_id, song_id: song.song_id})}>选择并开始翻唱</button>
        </div>
      </div>}
      <details><summary>使用本地歌曲</summary><PathField name="服务电脑上的音频路径" value={local} onChange={setLocal} kind="audio"/>
        <button disabled={!local || !activeModel.model || Boolean(active(cover)) || pending} onClick={() => generate({local_source: local})}>翻唱本地歌曲</button></details>
    </section>
    {cover && <section>
      <h2>当前翻唱</h2><p>{label(cover.state)} {cover.error}</p>
      {['FAILED', 'INTERRUPTED'].includes(cover.state) && <button onClick={() => submit('/covers/' + cover.id + '/retry')}>重试翻唱</button>}
      {audio && <>
        <CoverPlayer key={audio + (cover.result?.remix_id ?? '')} audio={audio + '?version=' + (cover.result?.remix_id ?? 'original')}
          cover={cover} submit={submit} jobs={jobs} openTraining={openTraining}/>
        <p>{cover.result?.song_title ?? cover.result?.song?.title ?? '本地歌曲'} · {cover.result?.inference?.settings?.inference_steps} 步 · 临时结果将在开始下一首时清理。</p>
        <div className="grid">
          <label>人声音量（dB）<input type="number" min="-40" max="12" value={vocal} onChange={e => setVocal(e.target.valueAsNumber)}/></label>
          <label>伴奏音量（dB）<input type="number" min="-40" max="12" value={backing} onChange={e => setBacking(e.target.valueAsNumber)}/></label>
        </div>
        <button onClick={() => submit('/covers/' + cover.id + '/remix', {vocal_gain_db: vocal, instrumental_gain_db: backing})}>重混音</button>
        <div className="grid">
          <label>作品名称<input value={title} onChange={e => setTitle(e.target.value)}/></label>
          <label>作品格式<select value={format} disabled={saving} onChange={e => setFormat(e.target.value as AudioFormat)}>
            <option value="wav">WAV（无损）</option>
            <option value="mp3">MP3（320 kbps）</option>
          </select></label>
        </div>
        <label className="check"><input type="checkbox" checked={stems} onChange={e => setStems(e.target.checked)}/>保留转换干声与伴奏（WAV）</label>
        <div className="actions"><button disabled={!title.trim() || saving} onClick={save}>{saving ? '保存中…' : '保存到作品库'}</button><a href={audio + '?download=true'}>下载 WAV</a></div>
      </>}
    </section>}
    <SavedLibrary saved={saved} models={models} reportError={reportError} refresh={refresh}/>
  </>;
}

function App() {
  const [page, setPage] = useState<'training' | 'cover'>('training');
  const [health, setHealth] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [requestedVersion, setRequestedVersion] = useState('');
  const [models, setModels] = useState<Model[]>([]);
  const [modelId, setModel] = useState('');
  const [speaker, setSpeaker] = useState('');
  const [dry, setDry] = useState('');
  const [dryStart, setDryStart] = useState(0);
  const [dryEnd, setDryEnd] = useState<number | null>(null);
  const [currentCover, setCurrentCover] = useState<Job | null>(null);
  const [enabled, setEnabled] = useState<{job?: Job; model?: Model}>({});
  const [saved, setSaved] = useState<Saved[]>([]);
  const [selected, setSelected] = useState(sessionStorage.getItem('timbreforge.job') ?? '');
  const [events, setEvents] = useState<string[]>([]);
  const [error, setError] = useState('');
  const [environment, setEnvironment] = useState<object>();
  const [points, setPoints] = useState<{file: string; step: number; epoch: number; reason: string}[]>([]);
  const [exports, setExports] = useState<{version_id: string; summary: {step: number; reason: string}}[]>([]);
  const refresh = () => {
    api<Job[]>('/jobs').then(setJobs).catch(e => setError(e.message));
    api<Dataset[]>('/datasets').then(setDatasets).catch(e => setError(e.message));
    api<Model[]>('/models').then(setModels).catch(e => setError(e.message));
    api<{job: Job | null}>('/covers/current').then(data => setCurrentCover(data.job)).catch(e => setError(e.message));
    api<{job?: Job; model?: Model}>('/models/enabled').then(setEnabled).catch(e => setError(e.message));
    api<Saved[]>('/saved-covers').then(setSaved).catch(e => setError(e.message));
  };
  useEffect(() => {
    api('/health').then(() => setHealth(true)).catch(e => setError(e.message));
    refresh();
    const timer = window.setInterval(refresh, 2500);
    return () => window.clearInterval(timer);
  }, []);
  useEffect(() => {
    setEvents([]);
    sessionStorage.setItem('timbreforge.job', selected);
    if (!selected) return;
    const stream = new EventSource('/api/jobs/' + selected + '/events');
    stream.onmessage = event => {
      const data = JSON.parse(event.data);
      setEvents(previous => [...previous.slice(-99), JSON.stringify(data.payload)]);
    };
    stream.addEventListener('end', () => {stream.close(); refresh();});
    return () => stream.close();
  }, [selected]);
  const submit: SelectJob = async (path, payload = {}) => {
    setError('');
    try {
      const job = await api<Job>(path, {...requestId(), ...payload});
      setSelected(job.id);
      refresh();
      return job;
    } catch (e) {setError((e as Error).message);}
  };
  const selectedJob = jobs.find(job => job.id === selected);
  const selectedJobState = selectedJob?.state;
  const selectedJobKind = selectedJob?.kind;
  useEffect(() => {
    setPoints([]);
    setExports([]);
    if (selectedJobKind !== 'train') return;
    const update = () => {
      api<typeof points>('/training/' + selected + '/checkpoints').then(setPoints).catch(e => setError(e.message));
      api<typeof exports>('/training/' + selected + '/exports').then(setExports).catch(e => setError(e.message));
    };
    update();
    const timer = window.setInterval(update, 3500);
    return () => window.clearInterval(timer);
  }, [selected, selectedJobState, selectedJobKind]);
  const model = models.find(row => modelKey(row) === modelId);
  const control = (action: string) => api('/jobs/' + selected + '/control', {...requestId(), action})
    .then(refresh).catch(e => setError(e.message));

  return <main>
    <header><div><h1>音色工坊</h1><p>TimbreForge</p></div><span className="status">{health ? '本地服务已连接' : '正在连接'}</span></header>
    <nav aria-label="业务模块">
      <button aria-current={page === 'training' ? 'page' : undefined} onClick={() => setPage('training')}>微调训练</button>
      <button aria-current={page === 'cover' ? 'page' : undefined} onClick={() => setPage('cover')}>歌曲翻唱</button>
    </nav>
    {error && <p role="alert">{error}</p>}
    <div hidden={page !== 'training'}>
      <Training datasets={datasets} submit={submit} selected={selectedJob} requestedVersion={requestedVersion}/>
      <ManagedTraining datasets={datasets} models={models} reportError={setError} refresh={refresh}/>
      <section><h2>模型效果测试</h2><ModelChooser models={models} model={model} speaker={speaker} setModel={setModel} setSpeaker={setSpeaker}/>
        <PathField name="本地干声" value={dry} onChange={setDry} kind="audio"/>
        <div className="grid">
          <label>开始（秒）<input type="number" min="0" step="0.1" value={dryStart} onChange={e => setDryStart(e.target.valueAsNumber)}/></label>
          <label>结束（留空使用原音频结尾）<input type="number" min="0.5" step="0.1" value={dryEnd ?? ''} onChange={e => setDryEnd(e.target.value ? e.target.valueAsNumber : null)}/></label>
        </div>
        <button disabled={!model || !speaker || !dry} onClick={() => model && submit('/model-tests', {
          model_id: model.model_id, version_id: model.version_id, speaker, path: dry, start_seconds: dryStart, end_seconds: dryEnd,
        })}>用 64 步生成试听</button>
        {selectedJob?.kind === 'test' && selectedJob.state === 'COMPLETED' && <div className="grid">
          <label>原干声<audio controls src={'/api/model-tests/' + selected + '/input'}/></label>
          <label>转换干声<audio controls src={'/api/model-tests/' + selected + '/converted'}/></label>
        </div>}
      </section>
    </div>
    <div hidden={page !== 'cover'}>
      <Covers models={models} model={model} speaker={speaker} choose={setModel} selectSpeaker={setSpeaker}
        submit={submit} activeModel={enabled} cover={currentCover} saved={saved} jobs={jobs}
        openTraining={version => {setRequestedVersion(version); setPage('training'); window.scrollTo(0, 0);}}
        reportError={setError} refresh={refresh}/>
    </div>
    <section>
      <h2>后台任务</h2>
      <label>任务<select value={selected} onChange={e => setSelected(e.target.value)}><option value="">选择任务</option>{jobs.map(job =>
        <option value={job.id} key={job.id}>{jobLabel(job.kind)} · {label(job.state)} · {job.id.slice(0, 8)}</option>
      )}</select></label>
      {selectedJob && <p>{label(selectedJob.state)} {selectedJob.error}</p>}
      {selectedJob?.kind === 'dataset_append' && selectedJob.result && <p>
        {selectedJob.result.role} · 新增 {selectedJob.result.added_clips} 段 · 训练片段 {selectedJob.result.previous_train_clips} → {selectedJob.result.train_clips}
      </p>}
      <TaskProgress progress={selectedJob?.latest_progress}/>
      <div className="actions">
        {active(selectedJob) && <button onClick={() => control('cancel')}>取消任务</button>}
        {selectedJob?.kind === 'train' && selectedJob.state === 'RUNNING' && <>
          <button onClick={() => control('pause')}>暂停并保存恢复点</button>
          <button onClick={() => control('export')}>导出模型并继续训练</button>
          <button onClick={() => control('stop_export')}>中止并导出模型</button>
        </>}
        {selectedJob?.kind === 'train' && ['PAUSED', 'INTERRUPTED', 'FAILED'].includes(selectedJob.state) &&
          <button onClick={() => submit('/training/' + selected + '/resume')}>继续训练</button>}
      </div>
      {points.length > 0 && <details><summary>训练恢复点</summary>
        <ul>{points.map(point => <li key={point.file}>step {point.step} · epoch {point.epoch} · {point.reason}{' '}
          {point.reason !== 'final' &&
            <button onClick={() => submit('/training/' + selected + '/derive', {file: point.file})}>从此恢复点派生</button>}
        </li>)}</ul>
      </details>}
      {exports.length > 0 && <details><summary>模型导出</summary>
        <ul>{exports.map(row => <li key={row.version_id}>step {row.summary.step} · {row.summary.reason}{' '}
          {!models.some(model => model.version_id === row.version_id) &&
            <button onClick={() => submit('/training/' + selected + '/exports/' + row.version_id)}>重试导出</button>}
        </li>)}</ul>
      </details>}
      {events.length > 0 && <pre aria-label="任务日志">{events.join('\n')}</pre>}
    </section>
    <section><details><summary>环境与资产</summary>
      <div className="actions">
        <button onClick={() => api<object>('/environment/doctor', {}).then(setEnvironment).catch(e => setError(e.message))}>运行环境诊断</button>
        <button onClick={() => submit('/assets/base', {source: null})}>准备官方底模</button>
        <button onClick={() => submit('/jobs/diagnostic')}>运行任务诊断</button>
      </div>
      {environment && <pre>{JSON.stringify(environment, null, 2)}</pre>}
    </details></section>
  </main>;
}

createRoot(document.getElementById('root')!).render(<App/>);
