window.addEventListener('error',function(e){console.error('[Studio]',e.error||e.message);});
const API='https://xori-training-api.onrender.com';
const state={page:'home',health:null,memory:null,diary:null,control:null,knowledge:null,messages:[],busy:false};
const titles={home:'Главная',chat:'Чат',training:'Обучение',errors:'Ошибки',tests:'Тесты',memory:'Память',knowledge:'Знания',diary:'Дневник',telegram:'Telegram',system:'Система',versions:'Версии',settings:'Настройки'};
const $=s=>document.querySelector(s);
async function api(path,options={}){
  const headers={Accept:'application/json',...(options.headers||{})};
  if(options.body && !headers['Content-Type']) headers['Content-Type']='application/json';
  const res=await fetch(API+path,{...options,headers});
  const data=await res.json().catch(()=>({}));
  if(!res.ok)throw new Error(data.error||'API '+res.status);
  return data;
}
