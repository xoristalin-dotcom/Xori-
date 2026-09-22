import fs from 'node:fs';
import path from 'node:path';

type Sample = [string, string];
type ClassDef = { id: string; reply: string };
type Weights = {
  version: 1;
  inputSize: number;
  hiddenSize: number;
  classes: string[];
  w1: number[][];
  b1: number[];
  w2: number[][];
  b2: number[];
};

const MODEL_DIR = path.join(process.cwd(), 'xori_neural');
const WEIGHTS_PATH = path.join(MODEL_DIR, 'weights.json');
const SEED_PATH = path.join(process.cwd(), 'xori_neural_seed.json');
const INPUT_SIZE = 256;
const HIDDEN_SIZE = 32;
const FEATURE_RE = /[\p{L}\p{N}]/u;

function hashFeature(s: string) {
  let h = 2166136261;
  for (const ch of s) { h ^= ch.charCodeAt(0); h = Math.imul(h, 16777619); }
  return (h >>> 0) % INPUT_SIZE;
}

function vectorize(text: string) {
  const s = text.toLowerCase().replace(/ё/g, 'е').replace(/[^\p{L}\p{N}\s]+/gu, ' ').replace(/\s+/g, ' ').trim();
  const x = new Array<number>(INPUT_SIZE).fill(0);
  const chars = Array.from(s);
  for (let i = 0; i < chars.length; i++) {
    if (FEATURE_RE.test(chars[i])) x[hashFeature(chars[i])] += 1;
    if (i + 1 < chars.length) x[hashFeature(chars[i] + chars[i + 1])] += 1.5;
    if (i + 2 < chars.length) x[hashFeature(chars[i] + chars[i + 1] + chars[i + 2])] += 2;
  }
  const norm = Math.sqrt(x.reduce((a,b)=>a+b*b,0)) || 1;
  return x.map(v=>v/norm);
}

function tanh(x:number){ return Math.tanh(x); }
function softmax(a:number[]) {
  const m=Math.max(...a); const e=a.map(v=>Math.exp(v-m)); const z=e.reduce((s,v)=>s+v,0);
  return e.map(v=>v/z);
}
function rand(seed:number){ let x=seed|0; return ()=>{ x=Math.imul(1664525,x)+1013904223; return ((x>>>0)/4294967296)-0.5; }; }

function freshWeights(classes:string[]):Weights {
  const r=rand(0x584f5249);
  const w1=Array.from({length:HIDDEN_SIZE},()=>Array.from({length:INPUT_SIZE},()=>r()*0.12));
  const w2=Array.from({length:classes.length},()=>Array.from({length:HIDDEN_SIZE},()=>r()*0.12));
  return {version:1,inputSize:INPUT_SIZE,hiddenSize:HIDDEN_SIZE,classes,w1,b1:new Array(HIDDEN_SIZE).fill(0),w2,b2:new Array(classes.length).fill(0)};
}

function forward(w:Weights,x:number[]) {
  const h=w.w1.map((row,j)=>tanh(row.reduce((s,v,i)=>s+v*x[i],w.b1[j])));
  return {h,p:softmax(w.w2.map((row,j)=>row.reduce((s,v,i)=>s+v*h[i],w.b2[j])))};
}

function train(w:Weights,samples:Sample[]) {
  const classIndex=new Map(w.classes.map((id,i)=>[id,i]));
  for(let epoch=0;epoch<70;epoch++){
    for(const [text,label] of samples){
      const y=classIndex.get(label); if(y===undefined) continue;
      const x=vectorize(text), f=forward(w,x), dz2=f.p.slice(); dz2[y]-=1;
      const dh=w.w2.map((row,j)=>row.reduce((s,v,i)=>s+v*dz2[j],0));
      const dz1=dh.map((v,j)=>v*(1-f.h[j]*f.h[j]));
      const lr=0.055;
      for(let j=0;j<w.w2.length;j++){ for(let i=0;i<HIDDEN_SIZE;i++) w.w2[j][i]-=lr*dz2[j]*f.h[i]; w.b2[j]-=lr*dz2[j]; }
      for(let j=0;j<HIDDEN_SIZE;j++){ for(let i=0;i<INPUT_SIZE;i++) w.w1[j][i]-=lr*dz1[j]*x[i]; w.b1[j]-=lr*dz1[j]; }
    }
  }
}

let cache: {weights:Weights, replies:Record<string,string>}|null=null;

function loadOrTrain(){
  if(cache) return cache;
  const seed=JSON.parse(fs.readFileSync(SEED_PATH,'utf8')) as {classes:ClassDef[],examples:Sample[]};
  let weights:Weights;
  try { weights=JSON.parse(fs.readFileSync(WEIGHTS_PATH,'utf8')); }
  catch {
    weights=freshWeights(seed.classes.map(c=>c.id));
    train(weights,seed.examples);
    fs.mkdirSync(MODEL_DIR,{recursive:true});
    fs.writeFileSync(WEIGHTS_PATH,JSON.stringify(weights));
  }
  cache={weights,replies:Object.fromEntries(seed.classes.map(c=>[c.id,c.reply]))};
  return cache;
}

export function generateXoriNeuralReply(text:string): {text:string; confidence:number; model:string}|null {
  if(!text.trim()) return null;
  const {weights,replies}=loadOrTrain();
  const {p}=forward(weights,vectorize(text));
  let best=0; for(let i=1;i<p.length;i++) if(p[i]>p[best]) best=i;
  const confidence=p[best];
  if(confidence<0.72) return null;
  const id=weights.classes[best];
  return {text:replies[id],confidence,model:'Xori Neural v0.1'};
}

export function getXoriNeuralStatus(){
  const {weights}=loadOrTrain();
  return {model:'Xori Neural v0.1',classes:weights.classes.length,inputSize:INPUT_SIZE,hiddenSize:HIDDEN_SIZE,weightsPath:WEIGHTS_PATH};
}
