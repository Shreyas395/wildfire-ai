const fileInput = document.getElementById('myFile');
const submitBtn = document.getElementById('submitBtn');
const previewImg = document.getElementById('preview');
const resultEl = document.getElementById('print_result');

// Preview selected image
fileInput.addEventListener('change', () => {
    const file = fileInput.files[0];
    if (!file) {
        previewImg.src = '';
        return;
    }

    const reader = new FileReader();
    reader.onload = e => {
        previewImg.src = e.target.result;
    };
    reader.readAsDataURL(file);
});

// Handle upload and prediction
submitBtn.addEventListener('click', async () => {
    const file = fileInput.files[0];
    if (!file) {
        resultEl.textContent = 'Please select an image first.';
        return;
    }

    resultEl.textContent = 'Uploading and analyzing...';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            resultEl.textContent = errorData.error || 'Error from server.';
            return;
        }

        const data = await response.json();
        const labelText = data.label || 'Unknown';
        const confidenceText =
            typeof data.confidence !== 'undefined' ? ` (confidence: ${data.confidence}%)` : '';
        resultEl.textContent = `${labelText}${confidenceText}`;
    } catch (err) {
        console.error(err);
        resultEl.textContent = 'Failed to contact server. Is it running?';
    }
});
