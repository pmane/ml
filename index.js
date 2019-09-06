
const webcamElement = document.getElementById('webcam');
const classifier = knnClassifier.create();

// Where to load the model from.
const MOBILENET_MODEL_TFHUB_URL =
    'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2'
// Size of the image expected by mobilenet.
const IMAGE_SIZE = 224;
// The minimum image size to consider classifying.  Below this limit the
// extension will refuse to classify the image.
const MIN_IMG_SIZE = 128;


let net;
async function app() {
  console.log('Loading mobilenet..');

  // Load the model.
  net = await mobilenet.load()
  //net = await loadModel();
  console.log('Sucessfully loaded model');

  await setupWebcam();

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = classId => {
    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(webcamElement, 'conv_preds');

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);
  };

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(webcamElement, 'conv_preds');
      // Get the most likely class and confidences from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['A', 'B', 'C'];
      document.getElementById('console').innerText = `
        prediction: ${classes[result.classIndex]}\n
        probability: ${result.confidences[result.classIndex]}
      `;
    }

    await tf.nextFrame();
  }
}



async function setupWebcam() {
  console.log('setting webcam');

  return new Promise((resolve, reject) => {
    const navigatorAny = navigator;
    navigator.getUserMedia = navigator.getUserMedia ||
        navigatorAny.webkitGetUserMedia || navigatorAny.mozGetUserMedia ||
        navigatorAny.msGetUserMedia;

    if (navigator.getUserMedia) {
      console.log('setting webcam 1');
      navigator.getUserMedia({video: true},
        stream => {
          console.log('setting webcam 2');
          webcamElement.srcObject = stream;
          webcamElement.addEventListener('loadeddata',  () => resolve(), false);
        },
        error => {
          console.log(error.name + ':' +error.message)
          //if you get an error NotReadableError:Could not start video source
          //then allow access to camera on your laptop setting - camera - allow access.
          return reject();
        });
    } else {
      reject();
    }
  });
}

/**
   * Loads mobilenet from URL and keeps a reference to it in the object.
   */
  async function loadModel() {
    console.log('Loading model....');
    const startTime = performance.now();
    try {
      net =
          await tf.loadGraphModel(MOBILENET_MODEL_TFHUB_URL, {fromTFHub: true});
      // Warms up the model by causing intermediate tensor values
      // to be built and pushed to GPU.
      tf.tidy(() => {
        net.predict(tf.zeros([1, IMAGE_SIZE, IMAGE_SIZE, 3]));
      });
      const totalTime = Math.floor(performance.now() - startTime);
      console.log(`Model loaded and initialized in ${totalTime} ms...`);
    } catch {
      console.error(
          `Unable to load model from URL: ${MOBILENET_MODEL_TFHUB_URL}`);
    }
    return net;
  }





app();