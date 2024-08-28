const toolbar = document.getElementById('toolbar');
const canvas = document.getElementById('drawing-board');
const ctx = canvas.getContext('2d');

canvas.width = 500;
canvas.height = 500;

ctx.strokeStyle = "#272774";

ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

/* Account for document format offsets */
const canvasOffsetHeight = canvas.offsetTop;
const canvasOffsetLeft = canvas.offsetLeft;

let isPainting = false;
let lineWidth = 30;
let startX;
let startY;

const handleClear = (ctx, canvas) => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById("prediction-value").innerHTML = "";
}

const handlePredict = (canvas) => {
    const image = canvas.toDataURL('image/png');
    fetch(
        "http://localhost:5000/predict",
        {
            method: "POST",
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: image
            })
        }
    ).then(async res => {
        data = await res.json();
        document.getElementById("prediction-value").innerHTML = data.digit;
    });
}

toolbar.addEventListener('click', e => {
    if (e.target.id === 'clear') {
        handleClear(ctx, canvas);
    } else if (e.target.id === 'predict') {
        handlePredict(canvas);
    }
});

const draw = (e) => {
    if (!isPainting) {
        return;
    }

    ctx.lineWidth = lineWidth;
    ctx.lineCap = 'round';

    ctx.lineTo(e.clientX - canvasOffsetLeft, e.clientY - canvasOffsetHeight);
    ctx.stroke();
}

canvas.addEventListener('mousedown', (e) => {
    isPainting = true;
    startX = e.clientX;
    startY = e.clientY;
});

canvas.addEventListener('mouseup', e => {
    isPainting = false;
    ctx.stroke();
    ctx.beginPath();
});

canvas.addEventListener('mousemove', draw);