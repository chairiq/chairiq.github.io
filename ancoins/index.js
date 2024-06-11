
//const MODEL_PATH ='http://83.212.171.115/deep-models/plants/tfjs_PFN/model.json';
const MODEL_PATH ='http://83.212.171.115/deep-models/web_model_ancoins_mobilenetv2/model.json';
const IMAGE_SIZE = 120;
const TOPK_PREDICTIONS = 12;

let model;
const loadModel = async () => {
  status('Please wait. Loading model...');

  model = await tf.loadLayersModel(MODEL_PATH);
  status('Model loaded!');
  $('#myCollapsible').collapse('toggle');
  $('#myCollapsible2').collapse('toggle');
  //console.table(model);

  // Warmup the model. This isn't necessary, but makes the first prediction
  // faster. Call `dispose` to release the WebGL memory allocated for the return
  // value of `predict`.
/*
  model.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3])).dispose();

  status('');

  // Make a prediction through the locally hosted cat.jpg.
  const catElement = document.getElementById('greek_salad');
  if (catElement.complete && catElement.naturalHeight !== 0) {
    predict(catElement);
    //catElement.style.display = '';
  } else {
    catElement.onload = () => {
      predict(catElement);
      //catElement.style.display = '';
    }
  }
  */
  document.getElementById('file-container').style.display = '';
  
};

/**
 * Given an image element, makes a prediction through mobilenet returning the
 * probabilities of the top K classes.
 */
 async function predict(imgElement2) {
  status('Predicting...');

  var imgElement = imgElement2.cloneNode(true);

  if(imgElement.width !== IMAGE_SIZE){
    imgElement.width = "120";
    imgElement.height = "120";  
  }
  // The first start time includes the time it takes to extract the image
  // from the HTML and preprocess it, in additon to the predict() call.
  const startTime1 = performance.now();
  // The second start time excludes the extraction and preprocessing and
  // includes only the predict() call.
  let startTime2;
  const logits = tf.tidy(() => {
    // tf.browser.fromPixels() returns a Tensor from an image element.
    const img = tf.browser.fromPixels(imgElement).toFloat();

    //const offset = tf.scalar(127.5);
    // Normalize the image from [0, 255] to [-1, 1].
    //const normalized = img.sub(offset).div(offset);
    const offset = tf.scalar(255);
    const normalized = img.div(offset);

    // Reshape to a single-element batch so we can pass it to predict.
    const batched = normalized.reshape([1, IMAGE_SIZE, IMAGE_SIZE, 3]);

    startTime2 = performance.now();
    // Make a prediction through mobilenet.
    return model.predict(batched);
    //return model.net.classify(batched);
  });
  // console.table(logits);
  // Convert logits to probabilities and class names.
  const classes = await getTopKClasses(logits, TOPK_PREDICTIONS);
  const totalTime1 = performance.now() - startTime1;
  const totalTime2 = performance.now() - startTime2;
  status(`Done in ${Math.floor(totalTime1)} ms ` +
    `(not including preprocessing: ${Math.floor(totalTime2)} ms)`);
  // Show the classes in the DOM.
  showResults(imgElement, classes);
}

/**
 * Computes the probabilities of the topK classes given logits by computing
 * softmax to get probabilities and then sorting the probabilities.
 * @param logits Tensor representing the logits from MobileNet.
 * @param topK The number of top predictions to show.
 */
 async function getTopKClasses(logits, topK) {
  const values = await logits.data();
  //console.log('number of clases '+values.length);
  const valuesAndIndices = [];
  for (let i = 0; i < values.length; i++) {
    valuesAndIndices.push({value: values[i], index: i});
  }

  valuesAndIndices.sort((a, b) => {
    return b.value - a.value;
  });

  const topkValues = new Float32Array(topK);
  const topkIndices = new Int32Array(topK);
  for (let i = 0; i < topK; i++) {
    topkValues[i] = valuesAndIndices[i].value;
    topkIndices[i] = valuesAndIndices[i].index;
  }

  const topClassesAndProbs = [];
  for (let i = 0; i < topkIndices.length; i++) {
    topClassesAndProbs.push({
      className: TARGET_CLASSES[topkIndices[i]],
      probability: topkValues[i]
    })
  }
  return topClassesAndProbs;
}

//
// UI
//

function showResults(imgElement, classes) {
  const predictionContainer = document.createElement('div');
  predictionContainer.className = 'pred-container';

  const imgContainer = document.createElement('div');
  imgContainer.appendChild(imgElement);
  predictionContainer.appendChild(imgContainer);

  const probsContainer = document.createElement('div');
  for (let i = 0; i < classes.length; i++) {
    const row = document.createElement('div');
    row.className = 'row2';

    const classElement = document.createElement('div');
    classElement.className = 'cell2';
    classElement.innerText = (i+1)+': '+classes[i].className;
    row.appendChild(classElement);

    const probsElement = document.createElement('div');
    probsElement.className = 'cell2';
    probsElement.innerText = classes[i].probability.toFixed(3);
    row.appendChild(probsElement);

    probsContainer.appendChild(row);
  }
  predictionContainer.appendChild(probsContainer);
  predictionsElement.insertBefore(predictionContainer, predictionsElement.firstChild);

  // for scrolling to results
  document.getElementById('system-status-head').scrollIntoView({ behavior: 'smooth'});
}


const filesElement = document.getElementById('files');
filesElement.addEventListener('change', evt => {
  document.getElementById('predictions').innerText = '';
  let files = evt.target.files;
  // Display thumbnails & issue call to predict each image.
  for (let i = 0, f; f = files[i]; i++) {
    // Only process image files (skip non image files)
    if (!f.type.match('image.*')) {
      continue;
    }
    let reader = new FileReader();
    const idx = i;
    // Closure to capture the file information.
    reader.onload = e => {
      // Fill the image & call predict.
      let img = document.createElement('img');
      img.src = e.target.result;
      img.width = IMAGE_SIZE;
      img.height = IMAGE_SIZE;
      img.onload = () => predict(img);
    };

    // Read in the image file as a data URL.
    reader.readAsDataURL(f);
  }
});

const demoStatusElement = document.getElementById('status');
const status = msg => demoStatusElement.innerText = msg;
const predictionsElement = document.getElementById('predictions');

loadModel();



document.getElementById("classesBTN").addEventListener("click", showClasses);
var model_body = document.getElementById("modal-body")

function showClasses(){
  const classesContainer = document.createElement('div');
  classesContainer.className = 'classes-container';

  const nameContainer = document.createElement('div');
  
  for (let i = 0; i < ObjectLength(TARGET_CLASSES); i++) {
    const rowX = document.createElement('div');
    rowX.className = 'row2';

    const classElement = document.createElement('div');
    classElement.className = 'cell2';
    classElement.innerText = (i+1)+': '+TARGET_CLASSES[i];
    rowX.appendChild(classElement);

    nameContainer.appendChild(rowX);    
  }
  classesContainer.appendChild(nameContainer);
  document.getElementById("modal-body").innerHTML = '';
  document.getElementById("modal-body").appendChild(classesContainer)
}


function ObjectLength( object ) {
    var length = 0;
    for( var key in object ) {
        if( object.hasOwnProperty(key) ) {
            ++length;
        }
    }
    return length;
};



var element1 = document.getElementById('img_pred1');
element1.addEventListener('click', function (){
  predict(element1);
}, true); 
/*var element2 = document.getElementById('img_pred2');
element2.addEventListener('click', function (){ 
  predict(element2);
}, true);
var element3 = document.getElementById('img_pred3');
element3.addEventListener('click', function (){ 
  predict(element3);
}, true);*/
var element4 = document.getElementById('img_pred4');
element4.addEventListener('click', function (){
  predict(element4);
}, true); 
/*var element5 = document.getElementById('img_pred5');
element5.addEventListener('click', function (){ 
  predict(element5);
}, true);*/
var element6 = document.getElementById('img_pred6');
element6.addEventListener('click', function (){ 
  predict(element6);
}, true);
var element7 = document.getElementById('img_pred7');
element7.addEventListener('click', function (){ 
  predict(element7);
}, true);
/*var element8 = document.getElementById('img_pred8');
element8.addEventListener('click', function (){ 
  predict(element8);
}, true);
var element9 = document.getElementById('img_pred9');
element9.addEventListener('click', function (){ 
  predict(element9);
}, true);*/
var element10 = document.getElementById('img_pred10');
element10.addEventListener('click', function (){ 
  predict(element10);
}, true);
/*var element11 = document.getElementById('img_pred11');
element11.addEventListener('click', function (){ 
  predict(element11);
}, true);
var element12 = document.getElementById('img_pred12');
element12.addEventListener('click', function (){ 
  predict(element12);
}, true);*/
