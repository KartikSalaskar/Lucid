import {useState} from 'react'
import Panels from './Panels'
import TrustPanel from './TrustPanel'

type Ind = {value:number;label:string}
type Res = {sha256:string;shape:number[];meta:Record<string,string>;processed:string;original:string;timings_ms:Record<string,unknown>;quality:{score:number;indicators:Record<string,Ind>};
  stats:{mean:number;std:number;entropy:number;edge_density:number;hist:number[]};
  prediction:{
  mode:string;
  label:string|null;
  confidence:number|null;
  note?:string;
  gradcam?:string;
  model?:Record<string,string>;
  uncertainty?:{
    entropy:number|null;
    uncertainty:number|null;
    level:string;
  };
};};

const Rows = ({o}:{o:Record<string,unknown>}) => <>{Object.entries(o).map(([k,v])=>
  <div className="row" key={k}><span>{k}</span><span>{String(v)}</span></div>)}</>

export default function App(){
  const [res,setRes]=useState<Res|null>(null), [err,setErr]=useState(''), [busy,setBusy]=useState(false)
  const [bri,setBri]=useState(100), [con,setCon]=useState(100), [zoom,setZoom]=useState(1), [inv,setInv]=useState(false)
  const [showProc,setShowProc]=useState(false)
  const [trustFile,setTrustFile]=useState<File|null>(null)

  async function upload(f:File){
    setTrustFile(f)
    setBusy(true); setErr(''); setRes(null)
    const fd=new FormData(); fd.append('file',f)
    try{
      const r=await fetch('/api/analyze',{method:'POST',body:fd}); const j=await r.json()
      if(!r.ok) throw new Error(j.detail ?? 'request failed')
      setRes(j)
    }catch(e){ setErr((e as Error).message) }
    setBusy(false)
  }
  const reset=()=>{setBri(100);setCon(100);setZoom(1);setInv(false)}
  const max=res? Math.max(...res.stats.hist):1

  return <div className="wrap">
    <h1>Medical Imaging Analysis &amp; Quality Platform</h1>
    <div className="warn">Research and educational software only. AI outputs are not medical diagnoses and should not be used as a substitute for professional clinical judgment.</div>
    <p><input type="file" accept=".png,.jpg,.jpeg,.dcm" onChange={e=>e.target.files?.[0]&&upload(e.target.files[0])}/> {busy&&'processing...'}</p>
    {err&&<p className="err">Error: {err}</p>}
    {res&&<div className="grid">
      <div className="card" style={{gridColumn:'1/-1'}}>
        <h2>Viewer | {showProc?'processed':'original'}</h2>
        <div className="view"><img src={showProc?res.processed:res.original} style={{
          transform:`scale(${zoom})`,filter:`brightness(${bri}%) contrast(${con}%) invert(${inv?1:0})`}}/></div>
        <p>brightness <input type="range" min={20} max={250} value={bri} onChange={e=>setBri(+e.target.value)}/>
        {' '}contrast <input type="range" min={20} max={300} value={con} onChange={e=>setCon(+e.target.value)}/>
        {' '}zoom <input type="range" min={1} max={5} step={.1} value={zoom} onChange={e=>setZoom(+e.target.value)}/>
        {' '}<button onClick={()=>setInv(!inv)}>invert</button> <button onClick={()=>setShowProc(!showProc)}>original/processed</button> <button onClick={reset}>reset</button></p>
      </div>
      <div className="card"><h2>Quality (baseline, not clinical)</h2>
        <div style={{fontSize:28}}>{res.quality.score}/100</div>
        {Object.entries(res.quality.indicators).map(([k,v])=><div className="row" key={k}>
          <span>{k} ({v.value})</span><b className={v.label}>{v.label}</b></div>)}</div>
      <div className="card"><h2>AI prediction | {res.prediction.mode}</h2>
        {res.prediction.label?<><div style={{fontSize:26}}>{res.prediction.label}</div><div>confidence {((res.prediction.confidence??0)*100).toFixed(1)}%
{res.prediction.uncertainty && res.prediction.uncertainty.uncertainty !== null && (
  <div>
    <div>
      Uncertainty: {(res.prediction.uncertainty.uncertainty * 100).toFixed(1)}%
    </div>
    <div>
      Uncertainty level: {res.prediction.uncertainty.level}
    </div>
  </div>
)}</div></>:<p>{res.prediction.note}</p>}
        {res.prediction.model&&<Rows o={res.prediction.model}/>}</div>
      {res.prediction.gradcam&&<div className="card"><h2>Regions contributing to model prediction (Grad-CAM)</h2>
        <div className="view"><img src={res.prediction.gradcam}/></div><p style={{color:'#8b98a5'}}>Model attribution only, not clinical evidence.</p></div>}
      <div className="card"><h2>Image stats</h2>
        <Rows o={{mean:res.stats.mean,std:res.stats.std,entropy:res.stats.entropy,edge_density:res.stats.edge_density}}/>
        <svg viewBox="0 0 320 80" width="100%">{res.stats.hist.map((h,i)=>
          <rect key={i} x={i*10} y={80-70*h/max} width={8} height={70*h/max} fill="#4aa3ff"/>)}</svg></div>
      <div className="card"><h2>Metadata (sanitized)</h2>
        <Rows o={{...res.meta,dimensions:res.shape.join('x'),sha256:res.sha256}}/></div>
      <div className="card"><h2>Timings (ms)</h2><Rows o={res.timings_ms}/></div>
    </div>}
    <TrustPanel file={trustFile}/>
    <h2 style={{marginTop:24}}>Model, dataset &amp; history</h2><Panels/>
  </div>
}