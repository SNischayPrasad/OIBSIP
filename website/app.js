// Runs the three exported models in the browser. Model parameters come from website/models/*.json,
// which build_models.py exports from the same training code as the notebooks. Each project page
// loads only the model whose demo it contains.

const $ = s => document.querySelector(s);
const load = name => fetch(`models/${name}.json`).then(r => r.json());
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function slider(container, { id, label, min, max, step, value, unit = '' }, onInput) {
  const wrap = document.createElement('label');
  wrap.innerHTML = `<span>${label}</span><input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}"><output></output>`;
  const input = wrap.querySelector('input'), out = wrap.querySelector('output');
  const sync = () => { out.textContent = Number(input.value).toFixed(step < 1 ? 1 : 0) + unit; onInput(); };
  input.addEventListener('input', sync);
  container.appendChild(wrap);
  return { input, set: v => { input.value = v; sync(); }, get: () => Number(input.value), sync };
}

/* ---------------- Task 1: Iris (KNN, k = 5) ---------------- */
if ($('#iris-sliders')) load('iris').then(m => {
  const colors = { setosa: css('--setosa'), versicolor: css('--versicolor'), virginica: css('--virginica') };
  const defs = [
    ['sepal length', 4.3, 7.9], ['sepal width', 2.0, 4.4], ['petal length', 1.0, 6.9], ['petal width', 0.1, 2.5],
  ];
  const box = $('#iris-sliders');
  const sliders = defs.map(([name, min, max], i) =>
    slider(box, { id: `iris-${i}`, label: name + ' (cm)', min, max, step: 0.1, value: [5.8, 3.0, 4.35, 1.3][i] }, predict));
  const Z = m.X.map(r => r.map((v, j) => (v - m.mean[j]) / m.scale[j]));

  // scatter of training flowers: petal length (x) vs petal width (y)
  const svg = $('#iris-plot'), W = 320, H = 220, P = 30;
  const sx = v => P + (v - 1) / (7 - 1) * (W - P - 10), sy = v => H - P - (v - 0) / (2.6 - 0) * (H - P - 10);
  let pts = `<line class="axis" x1="${P}" y1="${H - P}" x2="${W - 10}" y2="${H - P}"/><line class="axis" x1="${P}" y1="10" x2="${P}" y2="${H - P}"/>`;
  pts += `<text x="${W / 2}" y="${H - 6}" text-anchor="middle">petal length (cm)</text><text x="10" y="${H / 2}" transform="rotate(-90 10 ${H / 2})" text-anchor="middle">petal width (cm)</text>`;
  m.X.forEach((r, i) => { pts += `<circle cx="${sx(r[2])}" cy="${sy(r[3])}" r="3.2" fill="${colors[m.y[i]]}" fill-opacity=".55"/>`; });
  svg.innerHTML = pts + `<circle id="iris-you" r="7" fill="none" stroke="${css('--text')}" stroke-width="2.5"/>`;

  function predict() {
    const x = sliders.map(s => s.get());
    const z = x.map((v, j) => (v - m.mean[j]) / m.scale[j]);
    const nn = Z.map((r, i) => [Math.hypot(...r.map((v, j) => v - z[j])), m.y[i]]).sort((a, b) => a[0] - b[0]).slice(0, m.k);
    const votes = { setosa: 0, versicolor: 0, virginica: 0 };
    nn.forEach(([, c]) => votes[c]++);
    const best = Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
    $('#iris-pred').textContent = best[0].toUpperCase() + best.slice(1);
    $('#iris-pred').style.color = colors[best];
    $('#iris-votes').innerHTML = Object.entries(votes).map(([c, v]) =>
      `<div class="row"><span>${c}</span><div class="track"><div class="fill" style="width:${v / m.k * 100}%;background:${colors[c]}"></div></div><span>${v}/${m.k}</span></div>`).join('');
    const you = $('#iris-you');
    if (you) { you.setAttribute('cx', sx(x[2])); you.setAttribute('cy', sy(x[3])); }
  }
  document.querySelectorAll('[data-iris]').forEach(b => b.addEventListener('click', () =>
    b.dataset.iris.split(',').forEach((v, i) => sliders[i].set(v))));
  sliders.forEach(s => s.sync());
});

/* ---------------- Task 4: Spam (TF-IDF + Multinomial Naive Bayes) ---------------- */
if ($('#spam-text')) load('spam').then(m => {
  const stop = new Set(m.stopwords);
  const punct = /[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]/g;
  const clean = t => t.toLowerCase()
    .replace(/(https?:\/\/\S+|www\.\S+)/g, ' urltoken ')
    .replace(/\S+@\S+/g, ' emailtoken ')
    .replace(/\d{5,}/g, ' numtoken ')
    .replace(punct, ' ')
    .replace(/\d+/g, ' ')
    .split(/\s+/).filter(w => w.length > 1 && !stop.has(w));

  const text = $('#spam-text');
  function predict() {
    const toks = clean(text.value);
    if (!toks.length) {
      $('#spam-pred').textContent = '–'; $('#spam-pred').style.color = '';
      $('#spam-meter').style.width = '0'; $('#spam-prob').textContent = 'Type a message to classify it.';
      $('#spam-words').innerHTML = ''; return;
    }
    const grams = toks.concat(toks.slice(1).map((w, i) => toks[i] + ' ' + w));
    const counts = {};
    grams.forEach(g => { if (m.terms[g]) counts[g] = (counts[g] || 0) + 1; });
    const w = Object.fromEntries(Object.entries(counts).map(([g, c]) => [g, (1 + Math.log(c)) * m.terms[g][0]]));
    const norm = Math.hypot(...Object.values(w)) || 1;
    const contrib = Object.entries(w).map(([g, v]) => [g, v / norm * m.terms[g][1]]);
    const score = m.prior + contrib.reduce((s, [, c]) => s + c, 0);
    const p = 1 / (1 + Math.exp(-score));
    const spam = score > 0;
    $('#spam-pred').textContent = spam ? 'Spam' : 'Not spam';
    $('#spam-pred').style.color = spam ? css('--spam') : css('--ham');
    $('#spam-meter').style.width = (p * 100).toFixed(1) + '%';
    $('#spam-meter').style.background = spam ? css('--spam') : css('--ham');
    $('#spam-prob').textContent = `Spam probability: ${(p * 100).toFixed(1)}%`;
    $('#spam-words').innerHTML = contrib.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 8)
      .map(([g, c]) => `<span class="${c > 0 ? 's' : 'h'}">${g} ${c > 0 ? '↑' : '↓'}</span>`).join('') ||
      '<span>none of these words were seen in training</span>';
  }
  text.addEventListener('input', predict);
  document.querySelectorAll('[data-spam]').forEach(b => b.addEventListener('click', () => { text.value = b.dataset.spam; predict(); }));
  text.value = document.querySelector('[data-spam]').dataset.spam;
  predict();
});

/* ---------------- Task 5: Sales (degree-2 polynomial regression) ---------------- */
if ($('#sales-sliders')) load('sales').then(m => {
  const names = ['TV', 'Radio', 'Newspaper'];
  const box = $('#sales-sliders');
  const sliders = names.map((n, i) => slider(box, {
    id: `sales-${i}`, label: `${n} spend ($k)`, min: 0, max: Math.ceil(m.ranges[n][1] / 10) * 10, step: 1, value: [150, 25, 20][i],
  }, predict));
  const f = x => m.intercept + m.terms.reduce((s, term, i) =>
    s + m.coef[i] * term.split(' ').reduce((p, part) => {
      const [n, pow] = part.split('^');
      return p * Math.pow(x[names.indexOf(n)], pow ? Number(pow) : 1);
    }, 1), 0);

  function predict() {
    const x = sliders.map(s => s.get());
    $('#sales-pred').textContent = f(x).toFixed(2);
    $('#sales-budget').textContent = `$${x.reduce((a, b) => a + b, 0)}k`;
    $('#sales-shift-out').textContent = '';
  }
  $('#sales-shift').addEventListener('click', () => {
    const [tv, radio, news] = sliders.map(s => s.get());
    if (!news) { $('#sales-shift-out').textContent = 'There is no newspaper budget to move.'; return; }
    const before = f([tv, radio, news]);
    const half = news / 2, maxR = Number(sliders[1].input.max);
    const toRadio = Math.min(half, maxR - radio), toTV = news - toRadio;
    sliders[0].set(tv + toTV); sliders[1].set(radio + toRadio); sliders[2].set(0);
    const after = f(sliders.map(s => s.get()));
    $('#sales-shift-out').textContent = `Same total budget, sales ${after >= before ? 'up' : 'down'} from ${before.toFixed(2)} to ${after.toFixed(2)} thousand units (${after >= before ? '+' : ''}${((after / before - 1) * 100).toFixed(1)}%).`;
  });
  document.querySelectorAll('[data-sales]').forEach(b => b.addEventListener('click', () =>
    b.dataset.sales.split(',').forEach((v, i) => sliders[i].set(v))));
  sliders.forEach(s => s.sync());
});
