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

  const handleSubmit = async (e) => {
    e.preventDefault();
    setTextWarning(null);

    if (tab === "Text") {
      const words = text.trim().split(/\s+/).filter(Boolean);
      if (words.length < MIN_TEXT_WORDS) {
        setTextWarning(`Please enter at least ${MIN_TEXT_WORDS} words for accurate analysis. (Currently: ${words.length} words)`);
        return;
      }
    }

    setLoading(true);
    setResult(null);
    setScanning(true);
    
    let res;
    try {
    if (tab === "Text") {
        res = await axios.post('http://localhost:8000/detect/text', 
          new URLSearchParams({text}), 
          {headers: {'Content-Type': 'application/x-www-form-urlencoded'}}
        );
    } else {
      const formData = new FormData();
        formData.append('file', file);
        res = await axios.post(`http://localhost:8000/detect/${tab.toLowerCase()}`, formData, 
          {headers: {'Content-Type': 'multipart/form-data'}}
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
    setFile(selectedFile);
    
    // Show preview
    if (selectedFile) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const preview = document.querySelector('.file-preview');
        if (preview) {
          preview.innerHTML = '';
          
          if (tab === "Image") {
            const img = document.createElement('img');
            img.src = e.target.result;
            img.style.display = 'block';
            preview.appendChild(img);
          } else if (tab === "Video") {
            const video = document.createElement('video');
            video.src = e.target.result;
            video.controls = true;
            video.style.display = 'block';
            preview.appendChild(video);
          } else if (tab === "Audio") {
            const audio = document.createElement('audio');
            audio.src = e.target.result;
            audio.controls = true;
            audio.style.display = 'block';
            preview.appendChild(audio);
          }
        }
      };
      reader.readAsDataURL(selectedFile);
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
              onClick={() => { setTab(t); setResult(null); setFile(null); setText(""); setTextWarning(null); }}
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
                  <button 
                    type="submit" 
                    className="cyber-btn analyze-btn" 
                    disabled={loading || !file}
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