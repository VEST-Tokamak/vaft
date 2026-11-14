// Small landing-page controller. The SVG player and data load only on activation.
(function(){
  if(window.VAFTAnimationLoaded)return;window.VAFTAnimationLoaded=true;
  const players=new Map();let epoch=0;
  function cleanup(){epoch++;for(const p of players.values())p.controller.abort();players.clear();document.querySelectorAll('[data-animation]').forEach(r=>{r.dataset.playing='false';delete r.dataset.state;r.querySelectorAll('[data-state-download]').forEach(a=>a.hidden=true);const b=r.querySelector('[data-activate]');if(b){b.disabled=false;b.textContent='Interactive';}});}
  window.VAFTAnimationCleanup=cleanup;
  document.addEventListener('click',async event=>{
    const button=event.target.closest('[data-activate]');if(!button)return;
    const root=button.closest('[data-animation]');if(players.has(root))return;
    const controller=new AbortController(),record={controller,api:null};players.set(root,record);const activation=epoch;
    button.disabled=true;button.textContent='Loading…';const status=root.querySelector('[data-animation-status]');status.textContent='Loading scientific states…';
    try{
      const module=await import(root.dataset.player);
      if(controller.signal.aborted||activation!==epoch)return;
      record.api=await module.mount(root,{signal:controller.signal,onPlay:()=>{for(const [other,p] of players)if(other!==root)p.api?.pause();}});
      if(controller.signal.aborted||activation!==epoch)return;
      root.querySelectorAll('[data-state-download]').forEach(a=>a.hidden=false);
      status.textContent='Stored states loaded. Play or select a state.';button.textContent='Interactive loaded';
    }catch(error){
      if(error.name==='AbortError')return;
      controller.abort();players.delete(root);button.disabled=false;button.textContent='Retry interactive';status.textContent=`Playback unavailable: ${error.message}. The canonical poster and source remain available.`;
    }
  });
  function filter(root){const search=(root.querySelector('[data-animation-search]')?.value||'').toLowerCase();const family=root.querySelector('[data-animation-family]')?.value||'';const driver=root.querySelector('[data-animation-driver]')?.value||'';const capability=root.querySelector('[data-animation-capability]')?.value||'';let count=0;
    root.querySelectorAll('[data-animation-card]').forEach(card=>{const show=card.dataset.search.toLowerCase().includes(search)&&(!family||card.dataset.family===family)&&(!driver||card.dataset.driver===driver)&&(!capability||card.dataset.capability===capability);card.hidden=!show;if(show)count++;else players.get(card.querySelector('[data-animation]'))?.api?.pause();});const empty=root.querySelector('[data-animation-empty]');if(empty)empty.hidden=count!==0;
  }
  for(const type of ['input','change'])document.addEventListener(type,event=>{const root=event.target.closest('[data-animation-gallery]');if(root)filter(root);});
  window.addEventListener('pagehide',cleanup);
  if(typeof require==='function')require(['gitbook'],function(gitbook){gitbook.events.on('page.change',cleanup);});
})();
