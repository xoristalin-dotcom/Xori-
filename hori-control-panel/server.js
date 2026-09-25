import http from 'node:http';
import {URL} from 'node:url';
import fs from 'node:fs';
import path from 'node:path';

const port=Number(process.env.PORT||3000);
const state={name:'Хори',system:'Ты — Кёко Хори из Horimiya. Отвечай естественно, тепло и по-русски. Не управляй мыслями пользователя. Помни важные события диалога.',temperature:.75,topP:.9,maxTokens:900,model:'colab_train_xori_qwen25_3b_clean',lora:'hori-lora-v1',loraWeight:.8,hfUrl:process.env.HF_XORI_URL||'',renderUrl:process.env.RENDER_SERVICE_URL||'',colabUrl:process.env.COLAB_URL||'',githubRepo:process.env.GITHUB_REPO||'xoristalin-dotcom/Xori-'};
const configFile=path.join(process.cwd(),'hori-control-config.json');
try{if(fs.existsSync(configFile))Object.assign(state,JSON.parse(fs.readFileSync(configFile,'utf8')))}catch{}
const logs=[];
function log(x){logs.unshift({time:new Date().toISOString(),message:x});logs.splice(200);}
async function json(req){let s='';for await(const c of req)s+=c;return s?JSON.parse(s):{}}
async function apiFetch(url,opt={}){return fetch(url,opt)}
async function handler(req,res){
 const u=new URL(req.url,'http://localhost');
 if(req.method==='GET'&&u.pathname==='/api/health')return out(res,{ok:true,time:new Date().toISOString()});
 if(req.method==='GET'&&u.pathname==='/api/config'){log('GET /api/config');return out(res,{...state,secrets:{render:!!process.env.RENDER_API_TOKEN,github:!!process.env.GITHUB_TOKEN,hf:!!process.env.HF_TOKEN,cloudflare:!!process.env.CLOUDFLARE_API_TOKEN}})}
 if(req.method==='GET'&&u.pathname==='/api/logs')return out(res,logs);
 if(req.method==='PUT'&&u.pathname==='/api/config'){Object.assign(state,await json(req));try{fs.writeFileSync(configFile,JSON.stringify(state,null,2))}catch(e){log('Не удалось записать config: '+e.message)}log('Конфигурация сохранена кнопкой');return out(res,{ok:true,config:state})}
 if(req.method==='GET'&&u.pathname==='/api/render/status'){if(!process.env.RENDER_API_TOKEN)return out(res,{ok:false,error:'RENDER_API_TOKEN не задан'},400);try{const r=await apiFetch('https://api.render.com/v1/services',{headers:{Authorization:'Bearer '+process.env.RENDER_API_TOKEN}});const d=await r.text();log('Render API HTTP '+r.status);return out(res,{ok:r.ok,status:r.status,body:d.slice(0,5000)},r.ok?200:r.status)}catch(e){return out(res,{ok:false,error:e.message},500)}}
 if(req.method==='GET'&&u.pathname==='/api/github/repo'){if(!process.env.GITHUB_TOKEN)return out(res,{ok:false,error:'GITHUB_TOKEN не задан'},400);try{const r=await apiFetch('https://api.github.com/repos/'+state.githubRepo,{headers:{Authorization:'Bearer '+process.env.GITHUB_TOKEN,Accept:'application/vnd.github+json','User-Agent':'Hori-Control'}});const d=await r.json();log('GitHub API HTTP '+r.status);return out(res,{ok:r.ok,status:r.status,data:d},r.ok?200:r.status)}catch(e){return out(res,{ok:false,error:e.message},500)}}
 if(req.method==='GET'&&u.pathname==='/api/hf/status'){if(!state.hfUrl)return out(res,{ok:false,error:'HF_XORI_URL не задан'},400);try{const r=await apiFetch(state.hfUrl,{headers:process.env.HF_TOKEN?{Authorization:'Bearer '+process.env.HF_TOKEN}:{}});const t=await r.text();log('HF endpoint HTTP '+r.status);return out(res,{ok:r.ok,status:r.status,body:t.slice(0,4000)})}catch(e){return out(res,{ok:false,error:e.message},500)}}
 if(req.method==='POST'&&u.pathname==='/api/model/test'){if(!state.hfUrl)return out(res,{ok:false,error:'HF URL не задан'},400);try{const body=await json(req);const r=await apiFetch(state.hfUrl,{method:'POST',headers:{'Content-Type':'application/json',...(process.env.HF_TOKEN?{Authorization:'Bearer '+process.env.HF_TOKEN}:{})},body:JSON.stringify(body.payload||{prompt:'Скажи коротко: привет'})});const t=await r.text();log('Model test HTTP '+r.status);return out(res,{ok:r.ok,status:r.status,body:t})}catch(e){return out(res,{ok:false,error:e.message},500)}}
 if(req.method==='GET'&&u.pathname==='/'){res.writeHead(200,{'Content-Type':'text/html; charset=utf-8'});return res.end(page())}
 res.writeHead(404);res.end('Not found')
}
function out(res,d,status=200){res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store'});res.end(JSON.stringify(d))}
function page(){return `<!doctype html><html lang="ru"><head><meta name="viewport" content="width=device-width,initial-scale=1"><title>Hori Control</title><style>
*{box-sizing:border-box}body{margin:0;background:#090b10;color:#eee;font:14px system-ui,-apple-system,sans-serif}button,input,textarea{font:inherit}#app{display:flex;min-height:100vh}aside{width:245px;background:#0d0f14;border-right:1px solid #252832;padding:20px 12px;position:sticky;top:0;height:100vh}.brand{display:flex;gap:10px;align-items:center;padding:5px 8px 25px}.av{width:38px;height:38px;border-radius:12px;background:linear-gradient(135deg,#c6a8ff,#7657ff);display:grid;place-items:center;font-weight:800}.brand b{display:block;font-size:12px;letter-spacing:1.4px}.brand small{color:#777d8b}nav button{display:block;width:100%;border:0;background:none;color:#aeb3bf;text-align:left;padding:11px;border-radius:9px;margin:4px 0;cursor:pointer}nav button:hover,nav button.on{background:#1d1929;color:#ddd}.side{position:absolute;bottom:18px;color:#7c8290;font-size:11px;padding:8px}.dot{display:inline-block;width:7px;height:7px;background:#66d19e;border-radius:50%;margin-right:7px}main{width:min(1100px,100%);margin:auto;padding:30px}header{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px}.ey{font-size:10px;letter-spacing:2px;color:#8c849f}h1{font-size:27px;margin:5px 0 0}.save{border:0;border-radius:9px;padding:10px 14px;font-weight:700;cursor:pointer;background:#7657ff;color:white}.savebar{display:flex;gap:12px;align-items:center;margin-top:14px}.savebar span{font-size:11px;color:#777d8b}.grid{display:grid;gap:14px}.g3{grid-template-columns:repeat(3,1fr)}.g2{grid-template-columns:repeat(2,1fr)}.card{background:#13151c;border:1px solid #282c36;border-radius:14px;padding:18px;margin-bottom:14px}.card h3{font-size:14px;margin:0 0 15px}.big{font-weight:700;font-size:14px}.green{color:#72d5a0}.muted{color:#838998;font-size:12px;line-height:1.55}.flow{display:flex;gap:9px;flex-wrap:wrap;align-items:center}.flow b,.action{background:#171a22;border:1px solid #2b303a;border-radius:8px;padding:9px 11px;font-size:12px}.action{color:#ddd;cursor:pointer}.actions{display:flex;gap:8px;flex-wrap:wrap}label{display:block;color:#858b98;font-size:11px;margin-bottom:14px}input,textarea{width:100%;display:block;margin-top:7px;background:#0c0f14;border:1px solid #2b3039;border-radius:8px;color:#eee;padding:10px;outline:0}textarea{resize:vertical}.row{display:flex;justify-content:space-between;padding:11px 0;border-bottom:1px solid #242730;font-size:12px;color:#aab0bc}.row b{font-size:10px;color:#888}.term{background:#080a0d;border:1px solid #252933;border-radius:9px;padding:14px;min-height:220px;color:#9ce0bd;font:12px/1.8 monospace;white-space:pre-wrap}footer{text-align:center;color:#555b67;font-size:10px;padding:20px}@media(max-width:700px){aside{width:66px}.brand div:not(.av),nav button{font-size:0}.brand{padding-left:2px}main{padding:18px 12px}.g3,.g2{grid-template-columns:1fr}h1{font-size:22px}}
</style></head><body><div id="app"><aside><div class="brand"><div class="av">H</div><div><b>HORI CONTROL</b><small>central command</small></div></div><nav id="nav"></nav><div class="side"><span class="dot"></span><span id="status">Подключение…</span></div></aside><main><header><div><div class="ey">XORI / HORI / CONTROL</div><h1 id="title">Обзор</h1></div><button class="save" onclick="save()">💾 Сохранить раздел</button></header><div id="content"></div><footer>Hori Control v1 • секреты только на сервере</footer></main></div><script>
const tabs=[['overview','⌂ Обзор'],['hori','♡ Хори'],['model','◈ Модель'],['lora','◇ LoRA'],['render','▣ Render'],['colab','▤ Google Colab'],['github','◉ GitHub'],['logs','≋ Логи']];let tab='overview',s={};let logs=[];
const $=x=>document.getElementById(x);async function api(p,o={}){
 const ctrl=new AbortController();
 const timer=setTimeout(()=>ctrl.abort(),5000);
 try{
  const r=await fetch(p,{headers:{'Content-Type':'application/json'},...o,signal:ctrl.signal,cache:'no-store'});
  const text=await r.text();
  let d={}; try{d=text?JSON.parse(text):{}}catch{d={raw:text}}
  if(!r.ok)throw Error(d.error||'HTTP '+r.status);
  return d
 }catch(e){
  if(e.name==='AbortError') throw Error('Таймаут API (5 сек)');
  throw e
 }finally{clearTimeout(timer)}
}
function nav(){ $('nav').innerHTML=tabs.map(x=>'<button class="'+(x[0]===tab?'on':'')+'" onclick="tab=\''+x[0]+'\';render()">'+x[1]+'</button>').join('')}
function card(t,b){return '<section class="card"><h3>'+t+'</h3>'+b+'</section>'}
function saveBtn(label='💾 Сохранить изменения'){return '<div class="savebar"><button class="save" onclick="save()">'+label+'</button><span id="saved">Изменения не сохранены</span></div>'}function inp(k,label,extra=''){return '<label>'+label+'<input value="'+esc(s[k]??'')+'" oninput="s.'+k+'=this.value" '+extra+'></label>'}function esc(x){return String(x).replace(/&/g,'&amp;').replace(/"/g,'&quot;').replace(/</g,'&lt;')}
function render(){nav();$('title').textContent=tabs.find(x=>x[0]===tab)[1];let c='';
if(tab==='overview')c=card('Панель управления','<p class="muted">Здесь редактируются параметры Хори и подключения. Нажимай «Сохранить» — изменения отправляются на сервер панели. Автоматического применения к Render нет.</p><div class="actions"><button class="action" onclick="save()">💾 Сохранить всё</button><button class="action" onclick="load()">🔄 Обновить</button><button class="action" onclick="refreshLogs()">📋 Обновить логи</button></div>')+'<div class="grid g3">'+card('Сервер','<div class="big green">ONLINE</div><div class="muted">панель и API работают</div>')+card('Модель','<div class="big">'+esc(s.model)+'</div><div class="muted">активный inference</div>')+card('LoRA','<div class="big">'+esc(s.lora)+'</div><div class="muted">вес '+s.loraWeight+'</div>')+'</div>'+card('Архитектура','<div class="flow"><b>Telegram</b>→<b>Render bridge</b>→<b>Hori model</b>→<b>LoRA</b></div><p class="muted">Одна панель для параметров Хори, модели, LoRA, Render, Colab, GitHub и логов. Токены не попадают в браузер.</p>')+card('Интеграции','<div class="row"><span>Render</span><b>'+ (s.renderUrl?'CONNECTED':'NOT SET')+'</b></div><div class="row"><span>Hugging Face</span><b>'+ (s.hfUrl?'CONNECTED':'NOT SET')+'</b></div><div class="row"><span>GitHub</span><b>CONNECTED</b></div><div class="row"><span>Colab</span><b>'+ (s.colabUrl?'CONNECTED':'NOT SET')+'</b></div>');
if(tab==='hori')c=card('Личность Хори',inp('name','Имя')+'<label>System prompt<textarea rows="12" oninput="s.system=this.value">'+esc(s.system)+'</textarea></label><div class="grid g3">'+inp('temperature','Temperature','type="number" min="0" max="2" step=".05"')+inp('topP','Top P','type="number" min="0" max="1" step=".05"')+inp('maxTokens','Max tokens','type="number" min="1"')+'</div>'+saveBtn('💾 Сохранить параметры Хори'));
if(tab==='model')c=card('Основная модель',inp('model','Model ID')+inp('hfUrl','Inference / HF URL','placeholder="https://..."')+'<div class="actions"><button class="action" onclick="testHF()">🧪 Проверить модель</button><button class="action" onclick="testModel()">▶️ Тест генерации</button></div>'+saveBtn('💾 Сохранить модель'))+card('Fallback','<div class="row"><span>Qwen / Hori</span><b>ACTIVE</b></div><div class="row"><span>Cloudflare</span><b>FALLBACK</b></div><div class="row"><span>Pollinations</span><b>FALLBACK</b></div>');
if(tab==='lora')c=card('LoRA',inp('lora','Активная LoRA')+inp('loraWeight','Вес LoRA','type="number" min="0" max="2" step=".05"')+'<label>Источник<input id="loraSource" placeholder="username/repository"></label><div class="actions"><button class="action">🔄 Перезагрузить LoRA</button><button class="action">↩️ Подготовить откат</button></div>'+saveBtn('💾 Сохранить LoRA'));
if(tab==='render')c=card('Render',inp('renderUrl','Service URL','placeholder="https://...onrender.com"')+'<div class="actions"><button class="action" onclick="renderStatus()">🧪 Проверить Render API</button><button class="action" onclick="openURL(s.renderUrl||\'https://dashboard.render.com\')">↗ Открыть Render</button></div>'+saveBtn('💾 Сохранить Render'))+card('Секреты','<div class="row"><span>RENDER_API_TOKEN</span><b>SERVER ENV</b></div><div class="row"><span>GITHUB_TOKEN</span><b>SERVER ENV</b></div><div class="row"><span>HF_TOKEN</span><b>SERVER ENV</b></div><div class="row"><span>CLOUDFLARE_API_TOKEN</span><b>SERVER ENV</b></div>');
if(tab==='colab')c=card('Google Colab / обучение',inp('colabUrl','Notebook URL','placeholder="https://colab.research.google.com/..."')+'<div class="grid g3">'+inp('trainingBase','Base model')+inp('trainingEpochs','Epochs','type="number"')+inp('trainingBatch','Batch size','type="number"')+'</div><div class="actions"><button class="action" onclick="openURL(s.colabUrl)">↗ Открыть notebook</button></div>'+saveBtn('💾 Сохранить обучение')+'<p class="muted">Настройки обучения сохраняются в панели. Сам запуск Colab выполняется в Colab.</p>');
if(tab==='github')c=card('GitHub',inp('githubRepo','Репозиторий')+'<label>Ветка<input value="'+esc(s.githubBranch||'main')+'" oninput="s.githubBranch=this.value"></label><div class="actions"><button class="action" onclick="githubStatus()">🧪 Проверить GitHub API</button><button class="action" onclick="openURL(\'https://github.com/\'+s.githubRepo)">↗ Открыть репозиторий</button></div>'+saveBtn('💾 Сохранить GitHub'));
if(tab==='logs')c=card('Живые логи','<div class="actions"><button class="action" onclick="refreshLogs()">🔄 Обновить</button></div><br><div class="term">'+logs.map(x=>'['+new Date(x.time).toLocaleTimeString()+'] '+x.message).join('\n')+'</div>');
$('content').innerHTML=c}
async function load(){
 render();
 $('status').textContent='Проверяем API…';
 try{
  const cfg=await api('/api/config');
  s={...s,...cfg};
  $('status').textContent='Сервер подключён ✓';
  render();
 }catch(e){
  $('status').textContent='API недоступен: '+e.message;
  console.error('Hori Control API:',e);
 }
 refreshLogs();
}
async function save(){try{await api('/api/config',{method:'PUT',body:JSON.stringify(s)});$('status').textContent='Сохранено ✓';if($('saved'))$('saved').textContent='Сохранено '+new Date().toLocaleTimeString();refreshLogs()}catch(e){$('status').textContent=e.message}}
async function refreshLogs(){try{logs=await api('/api/logs');if(tab==='logs')render()}catch{}}
async function renderStatus(){try{let d=await api('/api/render/status');$('status').textContent='Render HTTP '+d.status;refreshLogs()}catch(e){$('status').textContent=e.message}}
async function githubStatus(){try{let d=await api('/api/github/repo');$('status').textContent='GitHub '+(d.data?.full_name||'OK');refreshLogs()}catch(e){$('status').textContent=e.message}}
async function testHF(){try{let d=await api('/api/hf/status');$('status').textContent='HF HTTP '+d.status;refreshLogs()}catch(e){$('status').textContent=e.message}}
async function testModel(){try{let d=await api('/api/model/test',{method:'POST',body:JSON.stringify({payload:{prompt:'Ответь одним коротким предложением: привет'}})});$('status').textContent='Model HTTP '+d.status;refreshLogs()}catch(e){$('status').textContent=e.message}}
function openURL(x){if(x)window.open(x,'_blank')}
load();setInterval(refreshLogs,10000);
</script></body></html>`}
http.createServer((req,res)=>handler(req,res).catch(e=>{log('ERROR '+e.message);out(res,{ok:false,error:e.message},500)})).listen(port,()=>log('Hori Control started on '+port));