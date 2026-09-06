const pages = document.querySelectorAll('.page');
const navItems = document.querySelectorAll('.nav-item');

function showPage(id) {
  pages.forEach(p => p.classList.toggle('active-page', p.id === id));
  navItems.forEach(n => n.classList.toggle('active', n.dataset.page === id));
  const titles = {dashboard:'Dashboard',screen:'Screen Resume',candidates:'Candidates',analytics:'Analytics',model:'AI Model'};
  document.getElementById('page-title').textContent = titles[id] || 'Dashboard';
  if (id === 'dashboard' || id === 'analytics') loadStats();
  if (id === 'candidates') loadCandidates();
}

navItems.forEach(btn => btn.addEventListener('click', () => showPage(btn.dataset.page)));
document.querySelectorAll('[data-go="screen"]').forEach(btn => btn.addEventListener('click', () => showPage('screen')));

async function loadStats() {
  const s = await fetch('/api/stats').then(r => r.json());
  document.getElementById('total').textContent = s.total;
  document.getElementById('shortlisted').textContent = s.shortlisted;
  document.getElementById('rejected').textContent = s.rejected;
  document.getElementById('review').textContent = s.needs_review;
  document.getElementById('avg-fit').textContent = `${Math.round(s.average_fit_score * 100)}%`;
  document.getElementById('avg-conf').textContent = `${Math.round(s.average_confidence * 100)}%`;
  document.getElementById('fit-bar').style.width = `${s.average_fit_score * 100}%`;
  document.getElementById('conf-bar').style.width = `${s.average_confidence * 100}%`;
  document.getElementById('a-fit').textContent = `${Math.round(s.average_fit_score * 100)}%`;
  document.getElementById('a-conf').textContent = `${Math.round(s.average_confidence * 100)}%`;
  document.getElementById('a-reviewed').textContent = s.human_reviewed;
  const max = Math.max(s.shortlisted, s.rejected, s.needs_review, 1);
  document.getElementById('bar-short').style.width = `${s.shortlisted / max * 100}%`;
  document.getElementById('bar-reject').style.width = `${s.rejected / max * 100}%`;
  document.getElementById('bar-review').style.width = `${s.needs_review / max * 100}%`;
}

async function loadCandidates() {
  const data = await fetch('/api/candidates').then(r => r.json());
  const body = document.getElementById('candidate-table');
  body.innerHTML = data.length ? data.map(c => {
    const cls = c.decision === 'SHORTLIST' ? 'green' : 'red';
    const statusCls = c.status === 'REVIEW' ? 'amber' : c.status === 'REVIEWED' ? 'green' : 'green';
    return `<tr><td><strong>${escapeHtml(c.candidate_name)}</strong></td><td>${Math.round(c.fit_score*100)}%</td><td>${Math.round(c.confidence*100)}%</td><td><span class="pill ${cls}">${c.decision}</span></td><td><span class="pill ${statusCls}">${c.status}</span></td><td>${c.created_at}</td></tr>`;
  }).join('') : '<tr><td colspan="6" style="text-align:center;color:#8b97a9;padding:30px">No candidates screened yet.</td></tr>';
}

function escapeHtml(value) { return String(value).replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch])); }

async function analyze() {
  const name = document.getElementById('candidate-name').value.trim() || 'Anonymous Candidate';
  const resume = document.getElementById('resume').value.trim();
  const job = document.getElementById('job').value.trim();
  if (!resume || !job) { alert('Please enter both the resume and job description.'); return; }

  const button = document.getElementById('analyze');
  button.disabled = true; button.textContent = 'Analyzing...';
  try {
    const response = await fetch('/score', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidate_name:name,resume_text:resume,job_description:job})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Screening failed');
    renderResult(data, name);
  } catch (err) { alert(err.message); }
  finally { button.disabled = false; button.textContent = '✦ Analyze Candidate'; }
}

document.getElementById('analyze').addEventListener('click', analyze);

function renderResult(d, name) {
  const result = document.getElementById('result');
  const decisionClass = d.decision === 'SHORTLIST' ? 'shortlist' : 'reject';
  const reviewRequired = d.review_required;
  result.className = 'result';
  result.innerHTML = `<div class="result-card"><div class="result-head"><div><span class="badge">AI ANALYSIS COMPLETE</span><h3>${escapeHtml(name)}</h3></div><span class="decision ${reviewRequired ? 'review' : decisionClass}">${reviewRequired ? '⚠ HUMAN REVIEW' : d.decision}</span></div><div class="result-grid"><div class="result-metric"><span>Resume–Job Fit</span><strong>${Math.round(d.fit_score*100)}%</strong></div><div class="result-metric"><span>Skill Overlap</span><strong>${d.skill_overlap}</strong></div><div class="result-metric"><span>AI Confidence</span><strong>${Math.round(d.confidence*100)}%</strong></div></div>${reviewRequired ? `<div class="review-box"><h4>⚠ Uncertainty detected</h4><p><p>The candidate was flagged for human review because the AI detected insufficient overall fit or uncertainty. A human should verify the result before making the final decision.</p><div class="review-actions"><button class="yes" onclick="saveReview(${d.candidate_id}, 'SHORTLIST')">✓ Human: Shortlist</button><button class="no" onclick="saveReview(${d.candidate_id}, 'REJECT')">✕ Human: Reject</button></div></div>` : `<div class="review-box" style="background:#f2f8ff;border-color:#dbe7f8"><h4>✓ Automatic decision</h4><p>Confidence is above the configured threshold, so the system accepted the AI recommendation automatically.</p></div>`}</div>`;
  result.scrollIntoView({behavior:'smooth',block:'start'});
}

async function saveReview(id, decision) {
  const response = await fetch('/api/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({candidate_id:id,human_decision:decision,notes:'Reviewed through HireAI dashboard.'})});
  if (!response.ok) { alert('Could not save review.'); return; }
  alert(`Human review saved: ${decision}`);
  loadStats();
  loadCandidates();
}

loadStats();
