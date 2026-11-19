/* Drawer accessibility, kept separate from scientific player state. */
require(['gitbook','jquery'],function(gitbook,$){
  if(window.VAFTAdaptiveLoaded)return;window.VAFTAdaptiveLoaded=true;
  const compact=window.matchMedia('(max-width:1240px)');let active=null,returnTo=null,observer=null;
  const book=()=>document.querySelector('.book');
  const menu=()=>document.querySelector('.book-header .fa-align-justify')?.closest('a,button');
  function panel(){return document.querySelector(active==='nav'?'.book-summary':'#page-toc');}
  function sync(which){
    const b=book();if(!b)return;
    const nav=b.classList.contains('with-summary'),toc=b.classList.contains('with-page-toc');
    if(which)active=which;
    if(active==='nav'&&!nav)active=toc?'toc':null;
    if(active==='toc'&&!toc)active=nav?'nav':null;
    if(!active)active=toc?'toc':nav?'nav':null;
    const open=compact.matches&&(nav||toc);
    if (b.classList.contains('va-drawer-open') !== open) b.classList.toggle('va-drawer-open',open);
    const inner=b.querySelector('.body-inner');if(inner)inner.inert=open;
    const summary=b.querySelector('.book-summary'),contents=b.querySelector('#page-toc');
    if(summary){summary.inert=!nav;summary.setAttribute('aria-hidden',summary.inert?'true':'false');}
    if(contents){contents.inert=compact.matches&&!toc;contents.setAttribute('aria-hidden',contents.inert?'true':'false');}
    const trigger=menu();if(trigger){trigger.setAttribute('aria-controls','va-main-navigation');trigger.setAttribute('aria-expanded',nav?'true':'false');}
  }
  window.VAFTDrawerSync=sync;
  function close(restore){const b=book();if(!b)return;b.classList.remove('with-summary','with-page-toc');$('.js-page-toc-toggle').attr('aria-expanded','false');active=null;sync();if(restore&&returnTo?.isConnected)returnTo.focus();}
  function init(){
    if(observer)observer.disconnect();active=null;returnTo=null;
    const b=book();if(!b)return;
    const summary=b.querySelector('.book-summary');if(summary){summary.id='va-main-navigation';summary.tabIndex=-1;summary.setAttribute('aria-label','Main navigation');}
    b.querySelectorAll('.va-drawer-backdrop').forEach(e=>e.remove());
    const backdrop=document.createElement('button');backdrop.type='button';backdrop.className='va-drawer-backdrop';backdrop.setAttribute('aria-label','Close navigation drawers');backdrop.tabIndex=-1;backdrop.addEventListener('click',()=>close(true));b.append(backdrop);
    if(compact.matches)close(false);else sync();
    observer=new MutationObserver(()=>sync());observer.observe(b,{attributes:true,attributeFilter:['class']});
  }
  document.addEventListener('click',event=>{
    if(!compact.matches)return;
    const trigger=event.target.closest('.book-header .btn');
    if(trigger&&trigger===menu()){
      event.preventDefault();event.stopImmediatePropagation();returnTo=trigger;
      const b=book(),open=!b.classList.contains('with-summary');b.classList.toggle('with-summary',open);sync(open?'nav':null);
      if(open)panel()?.focus({preventScroll:true});else trigger.focus();
    }else if(event.target.closest('.js-page-toc-toggle')){returnTo=document.querySelector('.page-toc-toggle');}
    else if(event.target.closest('.book-summary a')){close(false);}
  },true);
  document.addEventListener('keydown',event=>{
    if(!compact.matches||!book()?.classList.contains('va-drawer-open'))return;
    if(event.key==='Escape'){event.preventDefault();close(true);return;}
    if(event.key!=='Tab')return;
    const p=panel();if(!p)return;
    const list=[...p.querySelectorAll('a[href],button,input,select,[tabindex="0"]')].filter(e=>!e.disabled&&e.getClientRects().length);
    if(!list.length){event.preventDefault();p.focus();return;}
    const first=list[0],last=list[list.length-1];
    if(!p.contains(document.activeElement)||document.activeElement===p){event.preventDefault();(event.shiftKey?last:first).focus();}
    else if(event.shiftKey&&document.activeElement===first){event.preventDefault();last.focus();}
    else if(!event.shiftKey&&document.activeElement===last){event.preventDefault();first.focus();}
  });
  compact.addEventListener('change',()=>{if(compact.matches)close(false);else{const b=book();b?.classList.remove('with-page-toc');$('.js-page-toc-toggle').attr('aria-expanded','false');active=null;sync();}});
  gitbook.events.on('page.change',init);init();
});
