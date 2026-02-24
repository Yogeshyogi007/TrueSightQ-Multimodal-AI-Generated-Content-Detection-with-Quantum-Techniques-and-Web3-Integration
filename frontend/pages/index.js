import { useState } from 'react';
import axios from 'axios';

const TABS = ["Text", "Image", "Audio", "Video"];
const MIN_TEXT_WORDS = 30; // Minimum words required for reliable text analysis

export default function Home() {
  const [tab, setTab] = useState("Text");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [text, setText] = useState("");
  const [scanning, setScanning] = useState(false);
  const [textWarning, setTextWarning] = useState(null);
  const [fileWarning, setFileWarning] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [previewType, setPreviewType] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setTextWarning(null);
    setFileWarning(null);

    if (tab === "Text") {
      const words = text.trim().split(/\s+/).filter(Boolean);
      if (words.length < MIN_TEXT_WORDS) {
        setTextWarning(`Please enter at least ${MIN_TEXT_WORDS} words for accurate analysis. (Currently: ${words.length} words)`);
        return;
      }
      // Hard-cap text length before sending to backend to avoid network / OOM issues
      const maxChars = 4000;
      if (text.length > maxChars) {
        setTextWarning(`Input too long. Only the first ${maxChars} characters will be analyzed.`);
      }
    } else {
      if (!file) return;
      if (fileWarning) return;
    }

    setLoading(true);
    setResult(null);
    setScanning(true);
    
    let res;
    try {
      if (tab === "Text") {
        const maxChars = 4000;
        const payloadText = text.length > maxChars ? text.slice(0, maxChars) : text;
        res = await axios.post(
          'http://localhost:8000/detect/text',
          new URLSearchParams({ text: payloadText }),
          { headers: { 'Content-Type': 'application/x-www-form-urlencoded' } }
        );
      } else {
        const formData = new FormData();
        formData.append('file', file);
        res = await axios.post(
          `http://localhost:8000/detect/${tab.toLowerCase()}`,
          formData,
          { headers: { 'Content-Type': 'multipart/form-data' } }
        );
      }
    setResult(res.data);
    } catch (error) {
      console.error('Error:', error);
      setResult({ verdict: "Error", confidence: 0, modality: tab.toLowerCase() });
    } finally {
    setLoading(false);
      setScanning(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    setFileWarning(null);

    const expectedPrefix =
      tab === "Image" ? "image/" :
      tab === "Audio" ? "audio/" :
      tab === "Video" ? "video/" :
      null;

    if (selectedFile && expectedPrefix && selectedFile.type && !selectedFile.type.startsWith(expectedPrefix)) {
      setFile(null);
      setPreviewUrl(null);
      setPreviewType(null);
      setResult(null);
      setFileWarning("Please upload a valid file type.");
      // allow selecting the same file again
      e.target.value = "";
      return;
    }

    setFile(selectedFile);
    if (selectedFile) {
      // Use a blob URL for preview and track the modality
      const url = URL.createObjectURL(selectedFile);
      const type = tab.toLowerCase(); // "image" | "audio" | "video"
      setPreviewUrl(url);
      setPreviewType(type);
    } else {
      setPreviewUrl(null);
      setPreviewType(null);
    }
  };

  const startScanAnimation = () => {
    const scannerAnimation = document.querySelector('.scanner-animation');
    if (scannerAnimation) {
      scannerAnimation.style.display = 'block';
      setTimeout(() => {
        scannerAnimation.style.display = 'none';
      }, 3000);
    }
  };

  const getVerdictClass = (verdict) => {
    if (verdict?.toLowerCase().includes('ai')) return 'ai';
    if (verdict?.toLowerCase().includes('human')) return 'human';
    return 'uncertain';
  };

  return (
    <>
      {/* Matrix Code Rain Overlay */}
      <div className="matrix-code-rain"></div>
      {/* Scanline Overlay */}
      <div className="scanline-overlay"></div>
      {/* Matrix Background */}
      <div className="matrix-bg"></div>
      <div className="matrix-rain"></div>
      
      {/* Header */}
      <header className="header">
        <div className="logo">
          <h1>TrueSight<span>Q</span></h1>
        </div>
        <div className="nav-tabs">
        {TABS.map((t) => (
          <button
            key={t}
              className={`tab-btn ${tab === t ? 'active' : ''}`}
              onClick={() => {
                setTab(t);
                setResult(null);
                setFile(null);
                setText("");
                setTextWarning(null);
                setFileWarning(null);
                setPreviewUrl(null);
                setPreviewType(null);
              }}
          >
            {t}
          </button>
        ))}
      </div>
      </header>

      {/* Main Container */}
      <div className="container">
        {/* Scanner Section */}
        <div className="scanner-section">
          <div className="scanner-box">
            <div className="scanner-lights">
              <div className="light red"></div>
              <div className="light yellow"></div>
              <div className="light green"></div>
            </div>
            
            <div className="scanner-display">
              <div className="scanner-animation"></div>
              <div className="file-preview">
                {tab === "Text" ? (
                  <>
                    <i className="fas fa-keyboard"></i>
                    <span>ENTER TEXT TO ANALYZE</span>
                  </>
                ) : previewUrl && previewType === tab.toLowerCase() ? (
                  previewType === "image" ? (
                    <img
                      src={previewUrl}
                      alt="Preview"
                      style={{ display: 'block', maxWidth: '100%', maxHeight: '100%' }}
                    />
                  ) : previewType === "audio" ? (
                    <audio
                      src={previewUrl}
                      controls
                      style={{ display: 'block', width: '100%' }}
                    />
                  ) : (
                    <video
                      src={previewUrl}
                      controls
                      style={{ display: 'block', maxWidth: '100%', maxHeight: '100%' }}
                    />
                  )
                ) : tab === "Image" ? (
                  <>
                    <i className="fas fa-image"></i>
                    <span>UPLOAD IMAGE TO SCAN</span>
                  </>
                ) : tab === "Audio" ? (
                  <>
                    <i className="fas fa-microphone"></i>
                    <span>UPLOAD AUDIO TO ANALYZE</span>
                  </>
                ) : (
                  <>
                    <i className="fas fa-video"></i>
                    <span>UPLOAD VIDEO TO SCAN</span>
                  </>
                )}
              </div>
            </div>
            
            <div className="scanner-controls">
              {tab === "Text" ? (
                <form onSubmit={handleSubmit} style={{ width: '100%' }}>
                  <textarea
                    className="text-input"
                    placeholder="ENTER TEXT TO ANALYZE... (minimum 30 words for accurate results)"
                    value={text}
                    onChange={(e) => { setText(e.target.value); setTextWarning(null); }}
                    required
                  />
                  {textWarning && (
                    <div className="text-warning" role="alert">
                      {textWarning}
                    </div>
                  )}
                  <button 
                    type="submit" 
                    className="cyber-btn analyze-btn" 
                    disabled={loading || !text.trim()}
                    onClick={startScanAnimation}
                  >
                    {loading ? <div className="loading"></div> : <i className="fas fa-search"></i>}
                    {loading ? 'ANALYZING...' : 'ANALYZE TEXT'}
                  </button>
                </form>
              ) : (
                <form onSubmit={handleSubmit} style={{ width: '100%' }}>
                  <div className="file-upload">
                    <input
                      type="file"
                      accept={tab === "Image" ? 'image/*' : tab === "Audio" ? 'audio/*' : 'video/*'}
                      onChange={handleFileChange}
                      required
                    />
                    <div className="file-upload-text">
                      <i className={`fas fa-${tab === "Image" ? "image" : tab === "Audio" ? "microphone" : "video"}`}></i>
                      <span>CLICK TO UPLOAD {tab.toUpperCase()}</span>
                    </div>
                  </div>
                  <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', opacity: 0.8 }}>
                    {tab === "Image" && "Images: JPG/PNG, ~10MB or smaller recommended."}
                    {tab === "Audio" && "Audio: WAV/MP3/FLAC, short clips work best."}
                    {tab === "Video" && "Video: MP4 (H.264) recommended, short clips (≤2 minutes)."}
                  </div>
                  {fileWarning && (
                    <div className="text-warning" role="alert">
                      {fileWarning}
                    </div>
                  )}
                  <button 
                    type="submit" 
                    className="cyber-btn analyze-btn" 
                    disabled={loading || !file || !!fileWarning}
                    onClick={startScanAnimation}
                  >
                    {loading ? <div className="loading"></div> : <i className="fas fa-search"></i>}
                    {loading ? 'ANALYZING...' : `ANALYZE ${tab.toUpperCase()}`}
        </button>
      </form>
              )}
            </div>
          </div>
        </div>

        {/* Results Section */}
        <div className="results-section">
          <div className="results-container">
            {result && (
              <div className="result-card fade-in">
                <div className="card-title glitch" data-text={result.verdict}>
                  {result.verdict}
                </div>
                <div className="card-content">
                  <div className="info-item">
                    <span className="info-label">VERDICT:</span>
                    <span className={`verdict-badge ${getVerdictClass(result.verdict)}`}>
                      {result.verdict}
                    </span>
                  </div>
                  
                  <div className="info-item">
                    <span className="info-label">CONFIDENCE:</span>
                    <div style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
                      <div className="progress-bar">
                        <div 
                          className="progress-fill" 
                          style={{ width: `${(result.confidence * 100)}%` }}
                        ></div>
                      </div>
                      <span className="info-value">{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                  
                  <div className="info-item">
                    <span className="info-label">MODALITY:</span>
                    <span className="info-value">{result.modality?.toUpperCase()}</span>
                  </div>
                  
                  {result.perplexity && (
                    <div className="info-item">
                      <span className="info-label">AI SCORE:</span>
                      <span className="info-value">{(result.perplexity * 100).toFixed(1)}%</span>
                    </div>
                  )}
                </div>
        </div>
      )}
    </div>
        </div>
      </div>

      {/* Notification */}
      <div className={`notification ${result ? 'show' : ''} ${textWarning ? 'show warning' : ''}`}>
        <div className="notification-message">
          {textWarning ? textWarning : result ? 'ANALYSIS COMPLETE' : ''}
        </div>
      </div>
    </>
  );
} 