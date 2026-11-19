// Browser rendering only: every selected coordinate is a stored VAFT state.
const NS = 'http://www.w3.org/2000/svg';
const colours = {lcfs:'#172337',surface:'#a6a6a6',rational:'#1c67d2',island:'#1c67d2',separatrix:'#174f9f',passing:'#93bce8',o_point:'#1c67d2',x_point:'#163774',width:'#172337'};
const particleColour = style => /\belectron\b/i.test(style) ? '#1c67d2' : /\bion\b/i.test(style) ? '#d63027' : null;
const cache = new Map();
const svg = (name) => document.createElementNS(NS, name);
function check(data) {
  if (data.schema_version !== 1 || data.interpolation !== 'none' || !Array.isArray(data.states) || data.states.length < 2 || data.driver?.values?.length !== data.states.length || !data.driver.unit || !(data.fps > 0)) throw Error('Unsupported scientific-state artifact');
  if (!['phase','physical_time','parameter','process_stage','iteration'].includes(data.driver.kind)) throw Error('Unsupported scientific driver');
  if(data.coordinates?.system!=='scene_cartesian'||data.coordinates?.unit!=='cm'||data.coordinates?.dtype!=='float64'||data.driver.values.some(v=>typeof v!=='number'))throw Error('Unsupported scientific coordinates');
  const finite = (v) => { if (typeof v === 'number' && !Number.isFinite(v)) throw Error('Non-finite state'); if (v && typeof v === 'object') Object.values(v).forEach(finite); };
  finite(data);
  if (data.bounds.length !== 4 || data.bounds[2] <= data.bounds[0] || data.bounds[3] <= data.bounds[1]) throw Error('Invalid bounds');
  if (new Set(data.order).size !== data.order.length || !data.order.length) throw Error('Duplicate semantic IDs');
  for (const state of data.states) {
    const rows = [...data.static,...state];
    if (rows.length !== data.order.length || new Set(rows.map(r=>r.id)).size !== rows.length || rows.some(r=>!data.order.includes(r.id))) throw Error('State element mismatch');
    for (const r of rows) {
      const point = p => Array.isArray(p) && p.length === 2 && p.every(Number.isFinite);
      if (r.type === 'polyline' ? !Array.isArray(r.points) || !r.points.length || !r.points.every(point) : r.type === 'arrow' ? !point(r.start) || !point(r.end) : ['marker','label'].includes(r.type) ? !point(r.at) : true) throw Error('Unsupported geometry');
    }
  }
  if ((data.observables || []).some(o=>o.values.length !== data.states.length || !o.unit)) throw Error('Observable state mismatch');
}
async function artifact(url, hash, signal) {
  const key = `${url}:${hash}`;
  if (cache.has(key)) return cache.get(key);
  const response = await fetch(url, {signal});
  if (!response.ok) throw Error(`Scientific data unavailable (${response.status})`);
  const bytes = await response.arrayBuffer();
  if (hash) {
    const sum = [...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
    if (sum !== hash) throw Error('Scientific data integrity mismatch');
  }
  const data = JSON.parse(new TextDecoder().decode(bytes)); check(data);
  if (cache.size >= 2) cache.delete(cache.keys().next().value);
  cache.set(key,data); return data;
}
export async function mount(root, {signal, onPlay}) {
  const data = await artifact(root.dataset.artifact, root.dataset.sha256, signal);
  if (signal.aborted || !root.isConnected) throw new DOMException('Cancelled','AbortError');
  const panel = document.createElement('div'); panel.className = 'va-player';
  const stage = svg('svg'); stage.classList.add('va-stage'); stage.setAttribute('role','img'); stage.setAttribute('aria-label',`${data.builder}: stored scientific state`);
  const [x0,y0,x1,y1] = data.bounds;
  stage.setAttribute('viewBox',`${x0} ${-y1} ${x1-x0} ${y1-y0}`);
  const group = svg('g'); group.setAttribute('transform','scale(1,-1)'); stage.append(group);
  const defs = svg('defs'), tip = svg('marker'), triangle = svg('path');
  // Unique marker IDs prevent cross-player DOM collisions.
  tip.id = `va-tip-${crypto.randomUUID()}`; tip.setAttribute('viewBox','0 0 10 10'); tip.setAttribute('refX','9'); tip.setAttribute('refY','5'); tip.setAttribute('markerWidth','6'); tip.setAttribute('markerHeight','6'); tip.setAttribute('orient','auto-start-reverse');
  triangle.setAttribute('d','M 0 0 L 10 5 L 0 10 z'); triangle.setAttribute('fill','context-stroke'); tip.append(triangle);defs.append(tip);stage.prepend(defs);
  const nodes = new Map();
  const rows = new Map([...data.static,...data.states[0]].map(r=>[r.id,r]));
  function paint(row) {
    const charge = row.type === 'marker' && (row.role === 'ion_particle' ? '+' : row.role === 'electron_particle' ? '−' : null);
    let el = nodes.get(row.id);
    if (!el) {
      el = svg(charge ? 'g' : row.type === 'marker' ? (row.kind === 'x' ? 'path' : 'circle') : row.type === 'label' ? 'text' : 'path');
      el.dataset.element = row.id; el.dataset.role = row.role; nodes.set(row.id,el); group.append(el);
      el.setAttribute('stroke',particleColour(row.style) || colours[row.role] || '#334155');
      el.setAttribute('stroke-width',row.role === 'separatrix' ? '.027' : '.015');
      el.setAttribute('fill',row.type === 'marker' && row.kind !== 'x' ? el.getAttribute('stroke') : 'none');
      if (row.role === 'rational') el.setAttribute('stroke-dasharray','.09 .06');
      if (charge) {
        const ball=svg('circle');ball.setAttribute('r','.11');ball.setAttribute('fill',el.getAttribute('stroke'));
        const sign=svg('text');sign.textContent=charge;sign.setAttribute('transform','scale(1,-1)');sign.setAttribute('text-anchor','middle');sign.setAttribute('dominant-baseline','central');sign.setAttribute('font-family','sans-serif');sign.setAttribute('font-size','.17');sign.setAttribute('font-weight','700');sign.setAttribute('fill','white');sign.setAttribute('stroke','none');
        el.setAttribute('aria-label',charge==='+'?'Ion, positive charge':'Electron, negative charge');el.append(ball,sign);
      }
    }
    if (row.type === 'polyline') el.setAttribute('d',row.points.map((p,i)=>`${i?'L':'M'}${p[0]},${p[1]}`).join(' ') + (row.closed?' Z':''));
    else if (row.type === 'arrow') {
      el.setAttribute('d',`M${row.start} L${row.end}`); el.setAttribute('marker-end',`url(#${tip.id})`);
      if (row.both) el.setAttribute('marker-start',`url(#${tip.id})`);
    } else if (row.type === 'marker') {
      const [x,y] = row.at;
      if (charge) el.setAttribute('transform',`translate(${x},${y})`);
      else if (row.kind === 'x') el.setAttribute('d',`M${x-.07},${y-.07} L${x+.07},${y+.07} M${x-.07},${y+.07} L${x+.07},${y-.07}`);
      else {el.setAttribute('cx',x);el.setAttribute('cy',y);el.setAttribute('r',row.kind==='.'?'.02':'.06');}
    } else {
      el.setAttribute('transform',`translate(${row.at[0]},${row.at[1]}) scale(1,-1)`); el.setAttribute('font-size','.18'); el.setAttribute('fill','#172337'); el.setAttribute('stroke','none');
      el.textContent = row.text.replace(/\$/g,'');
    }
  }
  data.order.forEach(id=>paint(rows.get(id)));
  const controls = document.createElement('div'); controls.className = 'va-controls';
  const button = (label, fn) => {const b=document.createElement('button');b.type='button';b.textContent=label;b.addEventListener('click',fn,{signal});controls.append(b);return b;};
  let current=0, timer=null, speed=1, observer;
  const play = button('Play',()=>timer?pause():start());
  button('Previous',()=>{pause();select(Math.max(0,current-1));});
  button('Next',()=>{pause();select(Math.min(data.states.length-1,current+1));});
  button('Reset',()=>{pause();select(0);});
  const label=document.createElement('label');label.textContent='State ';const slider=document.createElement('input');slider.type='range';slider.min=0;slider.max=data.states.length-1;slider.step=1;slider.value=0;slider.setAttribute('aria-label',`${data.driver.name} stored state`);label.append(slider);controls.append(label);
  slider.addEventListener('input',()=>{pause();select(Number(slider.value));},{signal});
  const speedLabel=document.createElement('label');speedLabel.textContent='Presentation speed ';const speeds=document.createElement('select');speeds.setAttribute('aria-label','Presentation speed');
  for(const value of [.25,.5,1,2,4]){const option=document.createElement('option');option.value=value;option.textContent=`${value}×`;option.selected=value===1;speeds.append(option);}
  speeds.addEventListener('change',()=>{speed=Number(speeds.value);if(timer){pause();start();}},{signal});speedLabel.append(speeds);controls.append(speedLabel);
  const dialog=document.createElement('dialog');dialog.className='va-dialog';dialog.setAttribute('aria-label',`${data.builder} enlarged scientific replay`);
  const enlarge=button('Enlarge',()=>{
    if(dialog.open){dialog.close();return;}
    root.append(dialog);dialog.append(panel);panel.classList.add('va-enlarged');enlarge.setAttribute('aria-pressed','true');dialog.showModal();enlarge.focus();
  });enlarge.setAttribute('aria-pressed','false');
  dialog.addEventListener('close',()=>{if(!signal.aborted){panel.classList.remove('va-enlarged');root.append(panel);dialog.remove();enlarge.setAttribute('aria-pressed','false');enlarge.focus();}},{signal});
  const readout=document.createElement('output');readout.className='va-readout';
  panel.append(stage,readout,controls);
  const traces=[];
  for(const observable of data.observables || []){
    const title=document.createElement('p');title.className='va-trace-label';
    const plot=svg('svg');plot.classList.add('va-trace');plot.setAttribute('viewBox','0 0 600 100');plot.setAttribute('role','img');plot.setAttribute('aria-label',`${observable.name} (${observable.unit}) versus ${data.driver.name}`);
    const low=Math.min(...observable.values),high=Math.max(...observable.values),span=high-low||1;
    const line=svg('polyline');line.setAttribute('points',observable.values.map((v,i)=>`${10+580*i/(data.states.length-1)},${90-80*(v-low)/span}`).join(' '));line.setAttribute('fill','none');line.setAttribute('stroke',particleColour(observable.name)||'#1c67d2');line.setAttribute('stroke-width','1.5');
    const cursor=svg('line');cursor.setAttribute('y1',5);cursor.setAttribute('y2',95);cursor.setAttribute('stroke','#64748b');plot.append(line,cursor);panel.append(title,plot);traces.push({observable,title,cursor});
  }
  function select(index){current=index;slider.value=index;data.states[index].forEach(paint);root.dataset.state=index;readout.textContent=`${data.driver.name} = ${data.driver.values[index].toPrecision(7)} ${data.driver.unit} · state ${index+1}/${data.states.length} · ${data.fps} presentation fps`;
    for(const t of traces){const x=10+580*index/(data.states.length-1);t.cursor.setAttribute('x1',x);t.cursor.setAttribute('x2',x);t.title.textContent=`${t.observable.name} = ${t.observable.values[index].toPrecision(7)} ${t.observable.unit}`;}
  }
  function pause(){clearInterval(timer);timer=null;play.textContent='Play';root.dataset.playing='false';}
  function start(){onPlay();if(current===data.states.length-1)select(0);timer=setInterval(()=>{if(current===data.states.length-1)pause();else select(current+1);},1000/(data.fps*speed));play.textContent='Pause';root.dataset.playing='true';}
  document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();},{signal});
  document.addEventListener('keydown',e=>{if(e.key==='Escape'&&dialog.open){dialog.close();}},{signal});
  observer=new IntersectionObserver(entries=>{if(!entries[0].isIntersecting&&!dialog.open)pause();});observer.observe(root);
  signal.addEventListener('abort',()=>{pause();observer.disconnect();dialog.close();dialog.remove();panel.remove();},{once:true});
  root.append(panel);select(0);return {pause,select};
}
