function showPreview(inputId, imageId) {
  const input = document.getElementById(inputId);
  const image = document.getElementById(imageId);
  input.addEventListener('change', () => {
    const file = input.files[0];
    if (!file) return;
    image.src = URL.createObjectURL(file);
  });
}

function metricBox(title, value) {
  return `
    <div class="metric-box">
      <div class="metric-title">${title}</div>
      <div class="metric-value">${value}</div>
    </div>
  `;
}

showPreview('sar', 'sarPreview');
showPreview('optical', 'opticalPreview');

document.getElementById('runBtn').addEventListener('click', async () => {
  const sar = document.getElementById('sar').files[0];
  const optical = document.getElementById('optical').files[0];
  const status = document.getElementById('status');
  const results = document.getElementById('results');

  if (!sar || !optical) {
    status.textContent = 'Please upload both images first.';
    return;
  }

  const formData = new FormData();
  formData.append('sar_file', sar);
  formData.append('optical_file', optical);

  status.textContent = 'Running alignment...';
  results.classList.add('hidden');

  try {
    const res = await fetch('/api/align', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Alignment failed');

    document.getElementById('warpedImg').src = `${data.images.warped_sar}?t=${Date.now()}`;
    document.getElementById('overlayImg').src = `${data.images.overlay}?t=${Date.now()}`;
    document.getElementById('matchesImg').src = `${data.images.matches}?t=${Date.now()}`;

    document.getElementById('metrics').innerHTML = [
      metricBox('Method', data.method),
      metricBox('Raw matches', data.num_raw_matches),
      metricBox('Good matches', data.num_good_matches),
      metricBox('Inliers', data.num_inliers),
      metricBox('Mean confidence', data.mean_confidence),
      metricBox('Time (sec)', data.elapsed_sec),
    ].join('');

    status.textContent = data.message;
    results.classList.remove('hidden');
  } catch (err) {
    status.textContent = err.message;
  }
});
