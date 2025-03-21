<style>
*,
*:before,
*:after {
    box-sizing: border-box;
}

body{
    margin: 0;
    overflow: hidden;
}

canvas{
    border: solid white 5px;
    width: 50%;
    height: 100%;
}

.sidebar{
    height: 5vh;
    width: 100vw;
    float: left;
}

.container{
    width: 100vw;
    height: 95vh;
    float: left;
}

</style>
<body>
<div class="sidebar">
<button onclick="undo()">Undo</button>
<button onclick="removeAll()">Remove All</button>
<button onclick="reloadImages()">Reload Images</button>
<button onclick="savePoints()()">Save Points</button>
<head>Right click on an image to select a point, then right click a corresponding in the 
opposite image to establish a correspondence. When you are finished, click "Save Points"</head>
</div>
<div oncontextmenu="return false;" id="container" class="container">
<canvas id="left"></canvas><canvas id="right"></canvas>
</div>
</body>
<script src="https://code.jquery.com/jquery-3.5.0.js"></script>
<script>
var canvasLeft = document.getElementById("left"),
    contextLeft = canvasLeft.getContext("2d");

var canvasRight = document.getElementById("right"),
    contextRight = canvasRight.getContext("2d");

canvasLeft.width = canvasLeft.getBoundingClientRect().width - 10;
canvasLeft.height = canvasLeft.getBoundingClientRect().height - 10;

canvasRight.width = canvasRight.getBoundingClientRect().width - 10;
canvasRight.height = canvasRight.getBoundingClientRect().height - 10;

contextLeft.strokeStyle = "black";
contextLeft.lineWidth = 2;
contextRight.strokeStyle = "black";
contextRight.lineWidth = 2;

var img1 = new Image();
    img1.onload = initializeImageLeft;
    img1.src = '/image/left';

var img2 = new Image();
    img2.onload = initializeImageRight;
    img2.src = '/image/right';

function reloadImages() {
    d = new Date();
    t = d.getTime()
    img1.src = img1.src.split('?')[0] + '?' + t;
    img2.src = img2.src.split('?')[0] + '?' + t;
    console.log(img1.src);
    console.log(img2.src);
}

class Point {
    constructor(imgCoordX, imgCoordY, pairedPoint){
        this.imgCoordX = imgCoordX;
        this.imgCoordY = imgCoordY;
        this.pairedPoint = pairedPoint;
        if (this.pairedPoint != null){
            this.pairedPoint.pair = this;
        }
    }

    set pair(pt){
        this.pairedPoint = pt;
    }

    get pair(){
        return this.pairedPoint;
    }
    
    set X(x){
        this.imgCoordX = x;
    }
    
    get X(){
        return this.imgCoordX;
    }

    set Y(y){
        this.imgCoordY = y;
    }
    
    get Y(){
        return this.imgCoordY;
    }
}

leftPoints = [];
rightPoints = [];

leftTotalOffsetX = 0;
leftTotalOffsetY = 0;
rightTotalOffsetX = 0;
rightTotalOffsetY = 0;


function undo(){
    if (leftPoints.length > rightPoints.length){
        leftPoints.splice(leftPoints.length-1,1);
        drawLeftPoints(true);
    }
    else if (rightPoints.length > leftPoints.length){
        rightPoints.splice(rightPoints.length-1,1);
        drawRightPoints(true);
    }
    else if (leftPoints.length == rightPoints.length && leftPoints.length > 0){
        leftPoints.splice(leftPoints.length-1,1);
        rightPoints.splice(rightPoints.length-1,1);
        drawLeftPoints(true);
        drawRightPoints(true); 
    }
}

function removeAll(){
    leftPoints = [];
    rightPoints = [];
    drawLeftPoints(true);
    drawRightPoints(true);
}

function savePoints(){
    length = Math.min(leftPoints.length, rightPoints.length);
    leftOut = []
    rightOut = []
    for (let i = 0; i<length; i++){
        leftOut.push([leftPoints[i].X, leftPoints[i].Y]);
        rightOut.push([rightPoints[i].X, rightPoints[i].Y]);
    }
    data = {}
    data["leftPoints"] = leftOut;
    data["rightPoints"] = rightOut;
    $.ajax({
        type: "POST",
        url: "/save_points",
        data: JSON.stringify(data),
        contentType: "application/json"
    });

}

function exportPoints(){
    length = Math.min(leftPoints.length, rightPoints.length);
    leftOut = []
    rightOut = []
    for (let i = 0; i<length; i++){
        leftOut.push([leftPoints[i].X, leftPoints[i].Y]);
        rightOut.push([rightPoints[i].X, rightPoints[i].Y]);
    }
    data = {}
    data["leftPoints"] = leftOut;
    data["rightPoints"] = rightOut;
    $.ajax({
        type: "POST",
        url: "/points",
        data: JSON.stringify(data),
        contentType: "application/json"
    });

}

function initializeImageLeft(){
    leftTotalOffsetX = -(this.width/2 - canvasLeft.width/2);
    leftTotalOffsetY = -(this.height/2 - canvasLeft.height/2);
    contextLeft.drawImage(img1, leftTotalOffsetX,leftTotalOffsetY);
}
    
function initializeImageRight(){
    rightTotalOffsetX = -(this.width/2 - canvasRight.width/2);
    rightTotalOffsetY = -(this.height/2 - canvasRight.height/2);
    contextRight.drawImage(img2, rightTotalOffsetX,rightTotalOffsetY);
}

console.log(canvasLeft.width)
console.log(canvasLeft.height)

leftStartX = null;
leftStartY = null;
rightStartX = null;
rightStartY = null;
leftOffsetX = null;
leftOffsetY = null;
rightOffsetX = null;
rightOffsetY = null;
drawing = false;

canvasLeft.addEventListener("mousedown", mouseDown);
canvasLeft.addEventListener("mousemove", mouseMove);
canvasLeft.addEventListener("mouseup", stopDrawing);
canvasLeft.addEventListener("mouseleave", stopDrawing);
canvasRight.addEventListener("mousedown", mouseDown);
canvasRight.addEventListener("mousemove", mouseMove);
canvasRight.addEventListener("mouseup", stopDrawing);
canvasRight.addEventListener("mouseleave", stopDrawing);

function mouseDown(e){
    e.preventDefault();
    if (e.which == 3){
        if (e.target.id == "left"){
            if (leftPoints.length > rightPoints.length){
                leftPoints[leftPoints.length-1].X = e.offsetX+(-leftTotalOffsetX);
                leftPoints[leftPoints.length-1].Y = e.offsetY+(-leftTotalOffsetY);
            }
            else if (leftPoints.length < rightPoints.length){
                leftPoints.push(new Point(e.offsetX+(-leftTotalOffsetX), e.offsetY+(-leftTotalOffsetY), rightPoints[rightPoints.length-1]));
                drawRightPoints(true);
                exportPoints();
            }
            else{
                leftPoints.push(new Point(e.offsetX+(-leftTotalOffsetX), e.offsetY+(-leftTotalOffsetY), null));
            }
            drawLeftPoints(true);
        }
        else if (e.target.id == "right"){
            if (rightPoints.length > leftPoints.length){
                rightPoints[rightPoints.length-1].X = e.offsetX+(-rightTotalOffsetX);
                rightPoints[rightPoints.length-1].Y = e.offsetY+(-rightTotalOffsetY);
            }
            else if (rightPoints.length < leftPoints.length){
                rightPoints.push(new Point(e.offsetX+(-rightTotalOffsetX), e.offsetY+(-rightTotalOffsetY), leftPoints[leftPoints.length-1]));
                drawLeftPoints(true);
                exportPoints();
            }
            else{
                rightPoints.push(new Point(e.offsetX+(-rightTotalOffsetX), e.offsetY+(-rightTotalOffsetY), null));
            }
            drawRightPoints(true);
        }
        return false;
    }
    if (e.target.id == "left"){
        leftStartX = e.offsetX;
        leftStartY = e.offsetY;
    }
    else if (e.target.id == "right"){
        rightStartX = e.offsetX;
        rightStartY = e.offsetY;
    
    }
    drawing = true;
}

function mouseMove(e){
    if (drawing){
        if (e.target.id == "left"){
            leftOffsetX = e.offsetX - leftStartX + leftTotalOffsetX;
            leftOffsetY = e.offsetY - leftStartY + leftTotalOffsetY;
        }
        else if (e.target.id == "right"){
            rightOffsetX = e.offsetX - rightStartX + rightTotalOffsetX;
            rightOffsetY = e.offsetY - rightStartY + rightTotalOffsetY;
        }
        drawImage(e.target);
    }
}

function stopDrawing(e){
    if (drawing){
        drawing = false;
        if (e.target.id == "left"){
            leftTotalOffsetX = leftOffsetX;
            leftTotalOffsetY = leftOffsetY;
            drawLeftPoints(true);
        }
        else if (e.target.id == "right"){
            rightTotalOffsetX = rightOffsetX;
            rightTotalOffsetY = rightOffsetY;
            drawRightPoints(true);
        }
    }
}

function drawImage(canvas){
    if (canvas.id == "left"){
        contextLeft.clearRect(0,0, canvas.width, canvas.height);
        contextLeft.drawImage(img1, leftOffsetX, leftOffsetY);
        drawLeftPoints(false);
    }
    if (canvas.id == "right"){
        contextRight.clearRect(0,0, canvas.width, canvas.height);
        contextRight.drawImage(img2, rightOffsetX, rightOffsetY);
        drawRightPoints(false);
    }
}

function redrawImage(canvas){
    if (canvas.id == "left"){
        contextLeft.clearRect(0,0, canvas.width, canvas.height);
        contextLeft.drawImage(img1, leftTotalOffsetX, leftTotalOffsetY);
    }
    if (canvas.id == "right"){
        contextRight.clearRect(0,0, canvas.width, canvas.height);
        contextRight.drawImage(img2, rightTotalOffsetX, rightTotalOffsetY);
    }

}

function drawLeftPoints(redraw){
    if (redraw)
        redrawImage(canvasLeft);
    for(let i=0; i<leftPoints.length; i++){
        pt = leftPoints[i];
        if (pt.pair != null)
            contextLeft.fillStyle = "green";
        else
            contextLeft.fillStyle = "red";
        contextLeft.beginPath();
        if (leftOffsetX == null || leftOffsetY == null)
            contextLeft.arc(pt.imgCoordX+leftTotalOffsetX-1, pt.imgCoordY+leftTotalOffsetY-2,5,0,2*Math.PI);
        else
            contextLeft.arc(pt.imgCoordX+leftOffsetX-1, pt.imgCoordY+leftOffsetY-2,5,0,2*Math.PI);
        contextLeft.closePath();
        contextLeft.stroke();
        contextLeft.fill();
    }
}

function drawRightPoints(redraw){
    if (redraw)
        redrawImage(canvasRight);
    for(let i=0; i<rightPoints.length; i++){
        pt = rightPoints[i];
        if (pt.pair != null)
            contextRight.fillStyle = "green";
        else
            contextRight.fillStyle = "red";
        contextRight.beginPath();
        if (rightOffsetX == null || rightOffsetY == null)
            contextRight.arc(pt.imgCoordX+rightTotalOffsetX-1, pt.imgCoordY+rightTotalOffsetY-2,5,0,2*Math.PI);
        else
            contextRight.arc(pt.imgCoordX+rightOffsetX-1, pt.imgCoordY+rightOffsetY-2,5,0,2*Math.PI);
        contextRight.closePath();
        contextRight.stroke();
        contextRight.fill();
    }
}

</script>
