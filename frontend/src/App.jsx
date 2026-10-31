
import { useEffect, useRef, useState } from 'react';
import {
  CloudUpload,
  Image as ImageIcon,
  Zap,
  ShieldCheck,
  RotateCcw,
  Download,
  ArrowRight,
  Timer,
  Cpu,
  Layers,
  Info,
  AlertCircle,
  CheckCircle2,
} from 'lucide-react';

const formats = ['image/png', 'image/jpeg', 'image/webp'];

const models = {
  auto: {
    name: 'Automatic Selection',
    description:
      'MobileNetV3-Small predicts rain severity and selects PReNet6 or Restormer.',
  },
  prenet: {
    name: 'PReNet6',
    description: 'Manual Light-rain restoration.',
  },
  restormer: {
    name: 'Fine-tuned Restormer',
    description: 'Manual Heavy-rain restoration.',
  },
};

function Metric({ icon: Icon, label, value, suffix }) {
  return (
    <div className="metric">
      <span className="metric-icon">
        <Icon size={18} />
      </span>
      <div>
        <p>{label}</p>
        <strong>{value ?? '—'}</strong>
        {suffix && <small> {suffix}</small>}
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [maxSide, setMaxSide] = useState(256);
  const [selectedModel, setSelectedModel] = useState('auto');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [connected, setConnected] = useState(null);
  const [backendInfo, setBackendInfo] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const [split, setSplit] = useState(50);

  const chooser = useRef(null);

  useEffect(() => {
    fetch('/api/health')
      .then((response) => {
        if (!response.ok) throw new Error('Backend unavailable');
        return response.json();
      })
      .then((data) => {
        setConnected(true);
        setBackendInfo(data);
      })
      .catch(() => {
        setConnected(false);
        setBackendInfo(null);
      });
  }, []);

  useEffect(() => {
    if (!file) {
      setPreview('');
      return;
    }

    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function selectFile(candidate) {
    setError('');
    setResult(null);

    if (!candidate) return;

    if (!formats.includes(candidate.type)) {
      setError('Please upload a PNG, JPG or WEBP image.');
      return;
    }

    if (candidate.size > 12 * 1024 * 1024) {
      setError('Maximum upload size is 12 MB.');
      return;
    }

    setFile(candidate);
  }

  function changeModel(value) {
    if (loading) return;

    setSelectedModel(value);
    setResult(null);
    setError('');
  }

  async function restore() {
    if (!file || loading) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      if (
        !backendInfo?.automatic_routing_enabled ||
        backendInfo?.mode !== 'adaptive_restoration'
      ) {
        throw new Error(
          'The adaptive backend is not running. Restart Flask.'
        );
      }

      const payload = new FormData();
      payload.append('image', file);
      payload.append('model', selectedModel);
      payload.append('max_side', String(maxSide));

      const response = await fetch('/api/restore', {
        method: 'POST',
        body: payload,
      });

      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error('The backend returned an invalid response.');
      }

      if (!response.ok) {
        throw new Error(data.error || 'Processing failed.');
      }

      if (
        selectedModel === 'auto' &&
        data.mode !== 'automatic'
      ) {
        throw new Error('The backend did not execute automatic routing.');
      }

      if (
        selectedModel !== 'auto' &&
        data.mode !== `direct_${selectedModel}`
      ) {
        throw new Error('The backend executed an unexpected model.');
      }

      setResult(data);
      setSplit(50);
    } catch (err) {
      setError(err.message || 'Processing failed.');
    } finally {
      setLoading(false);
    }
  }

  function reset() {
    setFile(null);
    setResult(null);
    setError('');
    setSplit(50);

    if (chooser.current) {
      chooser.current.value = '';
    }
  }

  const heavySelected = selectedModel === 'restormer';

  const classification = result?.classification;
  const blocked = result?.status === 'blocked_uncertain';
  const completed = result?.status === 'restoration_completed';
  const predictedSeverity = classification?.predicted_class;
  const confidence = classification?.confidence;

  function confidenceText(value) {
    return typeof value === 'number'
      ? `${(value * 100).toFixed(2)}%`
      : '—';
  }

  return (
    <main className="app">
      <header className="header">
        <div className="brand">
          <span className="mark">
            <Layers size={21} />
          </span>
          <span>
            ADAPTIVE <b>VISION</b>
            <small>RAIN RESTORATION LAB</small>
          </span>
        </div>

        <div className="status">
          <span
            className={`status-dot ${
              connected === true
                ? 'online'
                : connected === false
                  ? 'offline'
                  : ''
            }`}
          />
          {connected === null
            ? 'Checking backend'
            : connected
              ? 'Backend connected'
              : 'Backend unavailable'}
        </div>
      </header>

      <section className="hero">
        <div className="eyebrow">
          <span className="pill">
            ADAPTIVE RAIN RESTORATION
          </span>
          <span>Computer Vision / Deep Learning</span>
        </div>

        <h1>
          See through <em>the rain.</em>
        </h1>

        <p>
          Upload a rainy image and let our severity classifier
          automatically select the restoration model, or
          manually choose Light or Heavy rain.
        </p>

        <div className="hero-points">
          <span><ShieldCheck size={15} /> Local processing</span>
          <span><Zap size={15} /> CUDA accelerated</span>
          <span><ImageIcon size={15} /> Adaptive restoration</span>
        </div>
      </section>

      <div className="workspace">
        <section className="panel upload-panel">
          <div className="section-head">
            <div>
              <span className="step">01 / INPUT</span>
              <h2>Rainy image</h2>
            </div>
            <button
              className="text-btn"
              onClick={reset}
              disabled={!file || loading}
            >
              <RotateCcw size={15} /> Reset
            </button>
          </div>

          <input
            ref={chooser}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            hidden
            onChange={(e) => selectFile(e.target.files?.[0])}
          />

          <div
            className={`dropzone ${dragOver ? 'over' : ''} ${
              preview ? 'has-image' : ''
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              selectFile(e.dataTransfer.files?.[0]);
            }}
            onClick={() => chooser.current?.click()}
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                chooser.current?.click();
              }
            }}
          >
            {preview ? (
              <>
                <img
                  className="uploaded-image"
                  src={preview}
                  alt="Selected rainy image"
                />
                <div className="image-overlay">
                  Click to change image
                </div>
              </>
            ) : (
              <div className="empty-upload">
                <span className="upload-icon">
                  <CloudUpload size={32} />
                </span>
                <strong>Drop your rainy image here</strong>
                <p>or click to browse your files</p>
                <small>PNG, JPG, WEBP · up to 12 MB</small>
              </div>
            )}
          </div>

          {file && (
            <div className="filename">
              <ImageIcon size={16} />
              <span title={file.name}>{file.name}</span>
              <small>{(file.size / 1048576).toFixed(2)} MB</small>
            </div>
          )}

          <div className="setting">
            <div>
              <label htmlFor="model-selection">
                Restoration mode
              </label>
              <p>Automatic or manual selection</p>
            </div>

            <select
              id="model-selection"
              value={selectedModel}
              onChange={(e) => changeModel(e.target.value)}
              disabled={loading}
            >
              <option value="auto">
                Automatic - Detect Rain Severity
              </option>
              <option value="prenet">
                Light Rain - PReNet6
              </option>
              <option value="restormer">
                Heavy Rain - Fine-tuned Restormer
              </option>
            </select>
          </div>

          <div className="notice">
            <Info size={16} />
            <span>
              <b>{models[selectedModel].name}</b>
              {' — '}
              {models[selectedModel].description}
            </span>
          </div>

          {!heavySelected && (
            <div className="setting">
              <div>
                <label htmlFor="resolution">
                  PReNet maximum processing dimension
                </label>
                <p>Used if PReNet is selected</p>
              </div>

              <select
                id="resolution"
                value={maxSide}
                onChange={(e) =>
                  setMaxSide(Number(e.target.value))
                }
                disabled={loading}
              >
                {[128, 256, 384, 512].map((size) => (
                  <option key={size} value={size}>
                    {size} px
                  </option>
                ))}
              </select>
            </div>
          )}

          {selectedModel !== 'prenet' && (
            <div className="notice">
              <Info size={16} />
              <span>
                <b>Heavy-rain restoration:</b>
                {' '}
                Restormer uses overlapping 128-pixel tiles
                at original resolution, with an initial
                1.2-megapixel limit.
              </span>
            </div>
          )}

          <button
            className="primary"
            onClick={restore}
            disabled={!file || loading || connected === false}
          >
            {loading ? (
              <>
                <span className="spinner" />
                Processing image...
              </>
            ) : (
              <>
                {selectedModel === 'auto'
                  ? 'Analyze and Restore'
                  : `Restore with ${models[selectedModel].name}`}
                <ArrowRight size={18} />
              </>
            )}
          </button>

          {error && (
            <div className="error">
              <AlertCircle size={17} />
              {error}
            </div>
          )}

          <div className="notice">
            <Info size={16} />
            <span>
              <b>Automatic safety gate.</b>
              {' '}
              Low-confidence predictions are blocked.
              You can then choose a model manually.
              PSNR and SSIM are unavailable without a clean reference.
            </span>
          </div>
        </section>

        <section className="panel results-panel">
          <div className="section-head">
            <div>
              <span className="step">02 / OUTPUT</span>
              <h2>Restoration result</h2>
            </div>
            {completed && (
              <span className="complete">
                <CheckCircle2 size={15} /> Completed
              </span>
            )}
          </div>

          {classification && (
            <div className="notice">
              <Info size={16} />
              <span>
                <b>Predicted severity: {predictedSeverity}</b>
                {' · '}
                Confidence: {confidenceText(confidence)}
                {' · '}
                Classification time: {
                  result.classification_seconds
                }s
                {classification.probabilities && (
                  <>
                    {' · '}
                    Light: {
                      confidenceText(
                        classification.probabilities.Light
                      )
                    }
                    {' · '}
                    Heavy: {
                      confidenceText(
                        classification.probabilities.Heavy
                      )
                    }
                  </>
                )}
              </span>
            </div>
          )}

          {blocked && (
            <div className="error">
              <AlertCircle size={19} />
              <div>
                <strong>Automatic restoration blocked.</strong>
                <p>{result.message}</p>
                <button
                  className="text-btn"
                  onClick={() => changeModel('prenet')}
                >
                  Select Light / PReNet
                </button>
                {' '}
                <button
                  className="text-btn"
                  onClick={() => changeModel('restormer')}
                >
                  Select Heavy / Restormer
                </button>
              </div>
            </div>
          )}

          {completed ? (
            <>
              <div className="notice">
                <CheckCircle2 size={16} />
                <span>
                  <b>Executed model:</b> {result.model}
                  {result.mode === 'automatic'
                    ? ' · Automatically selected'
                    : ' · Manually selected'}
                </span>
              </div>

              <div className="viewer">
                <div className="viewer-image">
                  <img
                    src={result.restored_url}
                    alt="Restored image"
                  />

                  <div
                    className="compare-overlay"
                    style={{
                      clipPath: `inset(0 ${100 - split}% 0 0)`,
                    }}
                  >
                    <img
                      src={result.input_url}
                      alt="Original rainy image"
                    />
                  </div>

                  <div
                    className="split-line"
                    style={{ left: `${split}%` }}
                  >
                    <span>↔</span>
                  </div>

                  <input
                    aria-label="Slide to compare images"
                    className="compare-slider"
                    type="range"
                    min="0"
                    max="100"
                    value={split}
                    onChange={(e) =>
                      setSplit(Number(e.target.value))
                    }
                  />
                </div>

                <div className="compare-labels">
                  <span>BEFORE · RAINY</span>
                  <span>AFTER · RESTORED</span>
                </div>
              </div>

              <div className="result-actions">
                <a
                  className="download"
                  href={result.restored_url}
                  download={`restored_${result.request_id}.png`}
                >
                  <Download size={17} />
                  Save restored image
                </a>

                <a
                  className="subtle-link"
                  href={result.input_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  View original
                </a>
              </div>

              <div className="metrics">
                <Metric
                  icon={Timer}
                  label="Inference time"
                  value={result.inference_seconds}
                  suffix="sec"
                />
                <Metric
                  icon={Cpu}
                  label="Allocated GPU memory"
                  value={result.peak_gpu_allocated_mib ?? 'CPU'}
                  suffix={
                    result.peak_gpu_allocated_mib != null
                      ? 'MiB'
                      : ''
                  }
                />
                <Metric
                  icon={ImageIcon}
                  label="Restored resolution"
                  value={result.output_size?.join(' × ')}
                />
                <Metric
                  icon={Layers}
                  label={
                    result.selected_model === 'restormer'
                      ? 'Tiles processed'
                      : 'Recurrent stages'
                  }
                  value={
                    result.selected_model === 'restormer'
                      ? result.tiles_processed
                      : result.recurrent_stages
                  }
                />
              </div>

              <p className="result-note">
                Original: {result.original_size?.join(' × ')}
                {' · '}
                Total: {result.total_processing_seconds}s
                {' · '}
                {result.device?.toUpperCase()}
              </p>

              {result.warning && (
                <div className="notice">
                  <AlertCircle size={16} />
                  <span>{result.warning}</span>
                </div>
              )}
            </>
          ) : !blocked ? (
            <div className="empty-result">
              <div className="empty-art">
                <span className="art-top" />
                <span className="art-bottom" />
                <ImageIcon size={40} />
              </div>

              <h3>
                {loading
                  ? 'Analyzing and restoring image'
                  : 'Your results appear here'}
              </h3>

              <p>
                {loading
                  ? 'Processing your image with the selected pipeline.'
                  : 'Upload a rainy image to view severity, restoration and timings.'}
              </p>
            </div>
          ) : null}
        </section>
      </div>

      <footer>
        <span>
          ADAPTIVE VISION INTELLIGENCE · B.TECH MAJOR PROJECT
        </span>
        <span>
          MobileNetV3-Small · PReNet6 · Fine-tuned Restormer
        </span>
      </footer>
    </main>
  );
}
