/*
Classifications
- Gyan Mudra
- Shuni Mudra
- Surya Mudra
- Buddhi Mudra
- Stop
*/

var audio;

const classifier = knnClassifier.create();
const webcamElement = document.getElementById('webcam');

let audioFile1 = document.getElementById('audio1');
let audioFile2 = document.getElementById('audio2');
let audioFile3 = document.getElementById('audio3');
let audioFile4 = document.getElementById('audio4');


let model;
let str;
var allText;

var sound_01 = new Howl({
  src: ['./audio/audio_01.wav'],
  format: ['wav']
});

var sound_02 = new Howl({
  src: ['./audio/audio_02.wav'],
  format: ['wav']
});

var sound_03 = new Howl({
  src: ['./audio/audio_03.wav'],
  format: ['wav']
});

var sound_04 = new Howl({
  src: ['./audio/audio_04.wav'],
  format: ['wav']
});

var uploaded_sound1, uploaded_sound2, uploaded_sound3, uploaded_sound4;



sound_01.volume(0.1);
sound_02.volume(0.1);
sound_03.volume(0.1);
sound_04.volume(0.1);

document.getElementById('start').addEventListener('click', () => sound.play());


function load() {
  // Load dataset file into a fresh classifier:
  classifier.setClassifierDataset( Object.fromEntries( JSON.parse(allText).map(([label, data, shape])=>[label, tf.tensor(data, shape)]) ) );

 }

 function readFile1(files) {
  var fileReader = new FileReader();
    fileReader.readAsArrayBuffer(files[0]);
    fileReader.onload = function(e) {
      //playAudioFile(e.target.result);
      console.log(("Filename: '" + files[0].name + "'"), ( "(" + ((Math.floor(files[0].size/1024/1024*100))/100) + " MB)" ));
      audioFile1.innerHTML = files[0].name + " uploaded to Gesture 1";

      let arrayBufferView1 = new Uint8Array(e.target.result);
      let blob1 = new Blob( [ arrayBufferView1 ], { type: 'music/mp3' } );
      let howlSource = URL.createObjectURL(blob1);

      uploaded_sound1 = new Howl({
        src: [howlSource],
        format: [ 'wav'],
      });

    }
}

function readFile2(files) {
  var fileReader = new FileReader();
    fileReader.readAsArrayBuffer(files[0]);
    fileReader.onload = function(e) {
      // playAudioFile(e.target.result);
      console.log(("Filename: '" + files[0].name + "'"), ( "(" + ((Math.floor(files[0].size/1024/1024*100))/100) + " MB)" ));
      audioFile2.innerHTML = files[0].name + " uploaded to Gesture 2";

      let arrayBufferView2 = new Uint8Array(e.target.result);
      let blob1 = new Blob( [ arrayBufferView2 ], { type: 'music/mp3' } );
      let howlSource = URL.createObjectURL(blob1);

      uploaded_sound2 = new Howl({
        src: [howlSource],
        format: [ 'wav'],
      });
    }
}

function readFile3(files) {
  var fileReader = new FileReader();
    fileReader.readAsArrayBuffer(files[0]);
    fileReader.onload = function(e) {
      // playAudioFile(e.target.result);
      console.log(("Filename: '" + files[0].name + "'"), ( "(" + ((Math.floor(files[0].size/1024/1024*100))/100) + " MB)" ));
      audioFile3.innerHTML = files[0].name + " uploaded to Gesture 3";

      let arrayBufferView3 = new Uint8Array(e.target.result);
      let blob1 = new Blob( [ arrayBufferView3 ], { type: 'music/mp3' } );
      let howlSource = URL.createObjectURL(blob1);

      uploaded_sound3 = new Howl({
        src: [howlSource],
        format: [ 'wav'],
      });
    }
}

function readFile4(files) {
  var fileReader = new FileReader();
    fileReader.readAsArrayBuffer(files[0]);
    fileReader.onload = function(e) {
      // playAudioFile(e.target.result);
      console.log(("Filename: '" + files[0].name + "'"), ( "(" + ((Math.floor(files[0].size/1024/1024*100))/100) + " MB)" ));
      audioFile4.innerHTML = files[0].name + " uploaded to Gesture 4";

      let arrayBufferView4 = new Uint8Array(e.target.result);
      let blob1 = new Blob( [ arrayBufferView4 ], { type: 'music/mp3' } );
      let howlSource = URL.createObjectURL(blob1);

      uploaded_sound4 = new Howl({
        src: [howlSource],
        format: [ 'wav'],
      });
    }
}

function playAudioFile(file) {
  var context = new window.AudioContext();
    context.decodeAudioData(file, function(buffer) {
      var source = context.createBufferSource();
        source.buffer = buffer;
        source.loop = false;
        source.connect(context.destination);
        source.start(0); 
    });
}

async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  model = await mobilenet.load();
  console.log('Successfully loaded model');

  // Create an object from Tensorflow.js data API which could capture image 
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = async classId => {
    // Capture an image from the web camera.
    const img = await webcam.capture();

    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = model.infer(img, true);

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);

    // Dispose the tensor to release the memory.
    img.dispose();
  };

  function save() {
    // Save it to a string:
    str = JSON.stringify( Object.entries(classifier.getClassifierDataset()).map(([label, data])=>[label, Array.from(data.dataSync()), data.shape]) );
    console.log("Dataset Saved");
  }

  // When clicking a button, add an example for that class.
  document.getElementById('class-gyan').addEventListener('click', () => addExample(0));
  document.getElementById('class-shuni').addEventListener('click', () => addExample(1));
  document.getElementById('class-surya').addEventListener('click', () => addExample(2));
  document.getElementById('class-buddhi').addEventListener('click', () => addExample(3));
  document.getElementById('class-nothing').addEventListener('click', () => addExample(4));

  //save dataset
  document.getElementById('save').addEventListener('click', () => save());

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      // Get the activation from mobilenet from the webcam.
      const activation = model.infer(img, 'conv_preds');
      // Get the most likely class and confidence from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['Gesture 1', 'Gesture 2', 'Gesture 3', `Gesture 4`, `Nothing`];
      document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;

      if(classes[result.label] == 'Gesture 1'){

        if(uploaded_sound1){
          uploaded_sound1.play();

          // Fires when the sound finishes playing.
          uploaded_sound1.once('end', function() {
            uploaded_sound1.play();
          });
        } else {
           sound_01.play();
          
        }

      } else if(classes[result.label] == 'Gesture 2') {
        if(uploaded_sound2){
          uploaded_sound2.play();
        } else {
          sound_02.play();
        }
      } else if(classes[result.label] == 'Gesture 3') {
        if(uploaded_sound3){
          uploaded_sound3.play();
        } else {
          sound_03.play();
        }
      } else if(classes[result.label] == 'Gesture 4') {
        if(uploaded_sound4){
          uploaded_sound4.play();
        } else {
          sound_04.play();
        }
      }

      // Dispose the tensor to release the memory.
      img.dispose();
    }

    await tf.nextFrame();
  } 
}

//download textfile
function download(filename, text) {
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}

// Start file download.
document.getElementById("dwn-btn").addEventListener("click", function(){
  // // Generate download of hello.txt file with some content
  // var text = document.getElementById("text-val").value;
 var filename = "dataset.txt";
  
  download(filename, str);
}, false);

//read textfile

function readTextFile(file)
{
    var rawFile = new XMLHttpRequest();
    rawFile.open("GET", file, false);
    rawFile.onreadystatechange = function ()
    {
        if(rawFile.readyState === 4)
        {
            if(rawFile.status === 200 || rawFile.status == 0)
            {
                allText = rawFile.responseText;
                
                //alert(allText);
            }
        }
    }
    rawFile.send(null);
}

document.getElementById("stop").addEventListener("click", function(){
  sound_01.stop();
  sound_02.stop();
  sound_03.stop();
  sound_04.stop();
  uploaded_sound1.stop();
  console.log("Stop all audio");


}, false);

//Read dataset
readTextFile("dataset.txt");
//Load dataset into classifier
load();
//Run classifier app
app();
