import 'dart:io';
import 'dart:core';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:async';
import 'package:permission_handler/permission_handler.dart';
import 'package:ffmpeg_kit_flutter/ffmpeg_kit.dart';
import 'package:google_speech/google_speech.dart';
import 'package:fftea/fftea.dart';
import 'package:xml/xml.dart' as xml;
import 'package:eneural_net/eneural_net.dart';
import 'package:flutter_sound_processing/flutter_sound_processing.dart';
import 'hlsttknn.dart';
import 'hlsttsvm.dart';
import 'hlsttann.dart';

const int bufferSize = 7839;
const int sampleRate = 16000;
const int hopLength = 350;
const int nMels = 40;
const int fftSize = 512;
const int mfcc = 40;

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Save Features',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: SoundRecorder(),
    );
  }
}

class SoundRecorder extends StatefulWidget {
  @override
  _SoundRecorderState createState() => _SoundRecorderState();
}


class _SoundRecorderState extends State<SoundRecorder> {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  StreamSubscription? _recorderSubscription;
  final speechToText = SpeechToText.viaApiKey("AIzaSyBgM6CzvKTZVEit_d-kOxjfstTW4OWe6ac");
  final config = RecognitionConfig(
    encoding: AudioEncoding.LINEAR16,
    model: RecognitionModel.command_and_search,
    enableAutomaticPunctuation: true,
    sampleRateHertz: 16000,
    languageCode: 'id-ID',
  );
  final List<String> keywords = ['tolong', 'tolong', 'help', 'help', 'aduh', 'aduh'];
  late Stream<Map<dynamic, dynamic>> result;
  bool showError = false;
  bool _isRecording = false;
  int recordingCount = 0;
  // int currentRecordingIndex = 0;
  String? _filePath;
  List<Map<String, dynamic>> trainingData = [];
  List<double> svmWeights = [];
  double svmBias = 0.0;
  List<double> bestWeights = [];
  double bestBias = 0.0;
  double bestLearningRate = 0.0;
  int bestEpoch = 0;
  double bestC = 0.0;
  double _amplitude = 0.0;
  // double _decibel = 0.0;
  bool _isAmplitudeHigh = false;
  bool _isAmplitudeLow = false;
  // String _sound = "Listening...";

  @override
  void initState() {
    super.initState();
    _initRecorder();
  }

  Future<void> _initRecorder() async {
    // Request permissions for microphone and storage
    var statusmic = await Permission.microphone.request();
    if (statusmic != PermissionStatus.granted) {
      print('Microphone permission not granted');
      return;
    }

    await _recorder.openRecorder();
    _recorder.setSubscriptionDuration(Duration(milliseconds: 2000));
  }
  // =================== MFCC ======================
  //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<String> _extractWaveform(String inputPath) async {
    String outputPath = '${inputPath}_waveform.pcm';
    String command = '-y -i "$inputPath" -ar 16000 -ac 1 -f s16le "$outputPath"';

    await FFmpegKit.execute(command).then((session) async {
      final returnCode = await session.getReturnCode();
      if (returnCode != null && returnCode.isValueSuccess()) {
        print('Waveform extracted successfully for $inputPath');
      } else {
        final output = await session.getOutput();
        print('Error extracting waveform: $output');
      }
    });

    return outputPath;
  }

  Future<List<int>> _getAudioBytes(String filePath) async {
    final audioFile = File(filePath);
    if (!await audioFile.exists()) {
      throw Exception("Audio file not found at path: $filePath");
    }
    final audioData = await audioFile.readAsBytes();
    return audioData;
  }

  List<double> normalizeAudioData(List<int> audioBytes) {
    List<double> normalizedData = [];
    for (int i = 0; i < audioBytes.length - 1; i += 2) {
      int sample = audioBytes[i] | (audioBytes[i + 1] << 8);
      if (sample > 32767) sample -= 65536;
      normalizedData.add(sample / 32768.0);
    }
    return normalizedData;
  }

  List<Float64List> melFilterbank(int numFilters, int fftSize, int sampleRate) {
    // Helper function to convert frequency to Mel scale
    double hzToMel(double hz) {
      return 2595 * log(1 + hz / 700) / ln10; // Convert Hz to Mel scale
    }

    // Helper function to convert Mel scale to frequency
    double melToHz(double mel) {
      return 700 * (pow(10, mel / 2595) - 1); // Convert Mel scale to Hz
    }

    // Create filterbank
    var melFilters = List<Float64List>.generate(numFilters, (i) => Float64List(fftSize ~/ 2 + 1));

    // Define the low and high frequency limits
    double lowFreqMel = hzToMel(0); // Lowest frequency (0 Hz)
    double highFreqMel = hzToMel(sampleRate / 2); // Nyquist frequency (half of sample rate)

    // Compute equally spaced Mel points
    var melPoints = List<double>.generate(numFilters + 2, (i) {
      return lowFreqMel + i * (highFreqMel - lowFreqMel) / (numFilters + 1);
    });

    // Convert Mel points back to Hz
    var hzPoints = melPoints.map(melToHz).toList();

    // Convert Hz points to FFT bin numbers
    var binPoints = hzPoints.map((hz) {
      return ((fftSize + 1) * hz / sampleRate).floor();
    }).toList();

    // Fill the Mel filterbank with triangular filters
    for (int i = 1; i < numFilters + 1; i++) {
      for (int j = binPoints[i - 1]; j < binPoints[i]; j++) {
        melFilters[i - 1][j] = (j - binPoints[i - 1]) / (binPoints[i] - binPoints[i - 1]);
      }
      for (int j = binPoints[i]; j < binPoints[i + 1]; j++) {
        melFilters[i - 1][j] = (binPoints[i + 1] - j) / (binPoints[i + 1] - binPoints[i]);
      }
    }

    return melFilters;
  }

  List<double> applyMelFilterbank(List<double> stftFrame, List<Float64List> melFilters) {
    var melEnergies = List<double>.filled(melFilters.length, 0.0);

    for (int i = 0; i < melFilters.length; i++) {
      melEnergies[i] = dot(melFilters[i], stftFrame);
    }

    return melEnergies;
  }

  double dot(List<double> vectorA, List<double> vectorB) {
    if (vectorA.length != vectorB.length) {
      throw Exception('Vector lengths must be equal for dot product');
    }

    double result = 0.0;
    for (int i = 0; i < vectorA.length; i++) {
      result += vectorA[i] * vectorB[i];
    }
    return result;
  }

  List<double> dct(List<double> input, int numCoefficients) {
    int N = input.length;
    List<double> output = List<double>.filled(numCoefficients, 0.0);

    for (int k = 0; k < numCoefficients; k++) {
      double sum = 0.0;
      for (int n = 0; n < N; n++) {
        sum += input[n] * cos((pi / N) * (n + 0.5) * k);
      }
      output[k] = sum;
    }

    return output;
  }

  List<double> computeMFCC(List<int> audioBytes, int sampleRate, int numCoefficients) {
    // Normalisasi data audio
    var audioSignal = normalizeAudioData(audioBytes);

    // Parameter untuk STFT
    final chunkSize = 512;
    final stft = STFT(chunkSize, Window.hanning(chunkSize));
    final spectrogram = <Float64List>[];

    // Jalankan STFT pada sinyal audio
    stft.run(audioSignal, (Float64x2List freq) {
      final magnitudes = freq.discardConjugates().magnitudes();
      spectrogram.add(magnitudes);
    });

    // Bangun filterbank Mel
    var melFilters = melFilterbank(26, chunkSize, sampleRate);

    // Mel spectrogram
    var melSpectrogram = <List<double>>[];
    for (var frame in spectrogram) {
      var melEnergies = applyMelFilterbank(frame, melFilters);
      melSpectrogram.add(melEnergies);
    }
    // Log energi Mel
    for (var i = 0; i < melSpectrogram.length; i++) {
      for (var j = 0; j < melSpectrogram[i].length; j++) {
        melSpectrogram[i][j] = log(melSpectrogram[i][j] + 1e-10);
      }
    }

    // Hitung MFCC untuk setiap frame
    var mfccFrames = <List<double>>[];
    for (var frame in melSpectrogram) {
      var dctCoeffs = dct(frame, numCoefficients);
      mfccFrames.add(dctCoeffs);
    }

    // Inisialisasi daftar untuk menyimpan rata-rata koefisien
    var averageMFCC = List<double>.filled(numCoefficients, 0.0);

    // Hitung rata-rata untuk setiap koefisien
    for (var i = 0; i < numCoefficients; i++) {
      double sum = 0.0;
      for (var frame in mfccFrames) {
        sum += frame[i];
      }
      averageMFCC[i] = sum / mfccFrames.length;
    }

    return averageMFCC;
  }

  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
  // ================================================

  // ============== EKSTRAKSI FITUR =================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<Map<String, dynamic>> extractAudioFeatures(String filePath) async {
    // Read the file
    File file = File(filePath);
    Uint8List audioData = await file.readAsBytes();

    // Convert Uint8List to List<int> (assuming 16-bit PCM)
    List<int> audioListInt = [];
    for (int i = 0; i < audioData.length; i += 2) {
      int value = (audioData[i + 1] << 8) | audioData[i];
      if (value >= 0x8000) value -= 0x10000;
      audioListInt.add(value);
    }

    // Convert to List<double>
    List<double> audioList = audioListInt.map((e) => e.toDouble()).toList();

    // Remove DC component (mean removal)
    double mean = audioList.reduce((a, b) => a + b) / audioList.length;
    audioList = audioList.map((v) => v - mean).toList();

    // Normalize amplitude
    double maxAbsVal = audioList.map((v) => v.abs()).reduce((a, b) => a > b ? a : b);
    if (maxAbsVal > 0) {
      audioList = audioList.map((v) => v / maxAbsVal).toList();
    }

    // MFCC Calculation using Flutter Sound Processing
    final flutterSoundProcessingPlugin = FlutterSoundProcessing();
    final signals = audioList.toList(); // Use the normalized audio list

    // Extract MFCC features
    final featureMatrix = await flutterSoundProcessingPlugin.getFeatureMatrix(
      signals: signals,
      fftSize: fftSize,
      hopLength: hopLength,
      nMels: nMels,
      mfcc: mfcc,
      sampleRate: sampleRate,
    );

    // Jika featureMatrix valid, hitung rata-rata koefisien MFCC
    List<double> averageMFCC = List.filled(40, 0.0); // Inisialisasi untuk 40 koefisien MFCC

    if (featureMatrix != null && featureMatrix.isNotEmpty) {
      int frameCount = featureMatrix.length ~/ 40; // Jumlah frame berdasarkan panjang data

      // Iterasi melalui setiap frame
      for (int frameIndex = 0; frameIndex < frameCount; frameIndex++) {
        for (int i = 0; i < 40; i++) {
          averageMFCC[i] += featureMatrix[frameIndex * 40 + i];
        }
      }

      // Menghitung rata-rata setiap koefisien MFCC
      averageMFCC = averageMFCC.map((mfcc) => mfcc / frameCount).toList();
    }

    // Return the extracted features as a map
    return {
      'mfcc': averageMFCC,
    };
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // ==================LOAD XML =====================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<void> loadFeaturesFromXml() async {
    // Tentukan lokasi file XML yang disimpan di penyimpanan eksternal
    final directory = await getExternalStorageDirectory();
    final filePath = '${directory?.path}/audio_features.xml';
    
    // Baca konten file XML dari lokasi penyimpanan eksternal
    final file = File(filePath);
    if (!file.existsSync()) {
      print('File tidak ditemukan: $filePath');
      return;
    }

    final xmlString = await file.readAsString();
    final xmlDoc = xml.XmlDocument.parse(xmlString);

    // Extract data from the XML
    for (var audioElement in xmlDoc.findAllElements('Audio')) {
      // Ubah label dari string menjadi integer
      int label = int.parse(audioElement.findElements('Label').first.text);
      List<double> mfcc = audioElement.findElements('MFCC')
          .map((e) => double.parse(e.text))
          .toList();

      trainingData.add({
        'label': label,  // Sekarang label disimpan sebagai integer
        'features': {
          'mfcc': mfcc,
        }
      });
    }

    print('Audio features loaded successfully.');
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // ============= Make XML ===================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<void> saveFeaturesToXml(List<Map<String, dynamic>> audioStore, String filePath) async {
    xml.XmlDocument xmlDocument;

    // Cek apakah file sudah ada
    final file = File(filePath);
    if (file.existsSync()) {
      final xmlString = await file.readAsString();
      xmlDocument = xml.XmlDocument.parse(xmlString);
    } else {
      // Membuat dokumen XML baru jika file belum ada
      xmlDocument = xml.XmlDocument([
        xml.XmlProcessing('xml', 'version="1.0"'),
        xml.XmlElement(xml.XmlName('AudioFeatures')),
      ]);
    }

    // Menambahkan fitur audio baru ke elemen root 'AudioFeatures'
    final rootElement = xmlDocument.rootElement;
    for (var audioData in audioStore) {
      final audioElement = xml.XmlElement(xml.XmlName('Audio'));

      audioElement.children.add(xml.XmlElement(xml.XmlName('Label'), [], [xml.XmlText(audioData['label'].toString())]));
      // audioElement.children.add(xml.XmlElement(xml.XmlName('Frequency'), [], [xml.XmlText(audioData['features']['frequency'].toString())]));
      // audioElement.children.add(xml.XmlElement(xml.XmlName('Amplitude'), [], [xml.XmlText(audioData['features']['amplitude'].toString())]));
      // audioElement.children.add(xml.XmlElement(xml.XmlName('Decibel'), [], [xml.XmlText(audioData['features']['decibel'].toString())]));
      // audioElement.children.add(xml.XmlElement(xml.XmlName('Amplitude_flutter_sound'), [], [xml.XmlText(audioData['features']['amplitude_flutter_sound'].toString())]));
      // audioElement.children.add(xml.XmlElement(xml.XmlName('Decibel_flutter_sound'), [], [xml.XmlText(audioData['features']['decibel_flutter_sound'].toString())]));
      List mfccValues = audioData['features']['mfcc'].toList();
      for (var mfcc in mfccValues) {
        audioElement.children.add(xml.XmlElement(xml.XmlName('MFCC'), [], [xml.XmlText(mfcc.toString())]));
      }

      rootElement.children.add(audioElement);
    }

    // Menyimpan dokumen XML yang sudah diperbarui ke file yang sama
    await file.writeAsString(xmlDocument.toXmlString(pretty: true));
    print('New audio features have been appended to the file at: $filePath');
  }

  Future<void> loadAndAppendFeaturesFromAssets(List<Map<String, dynamic>> audioStore, String filePath) async {
    for (int i = 1; i <= 14; i++) {
      // Muat file audio dari assets
      ByteData audioData = await rootBundle.load('assets/test$i.wav');

      // Simpan sementara sebagai file untuk proses ekstraksi
      final tempDir = await getTemporaryDirectory();
      final tempFile = File('${tempDir.path}/test$i.wav');
      await tempFile.writeAsBytes(audioData.buffer.asUint8List());

      // Ekstrak fitur audio
      Map<String, dynamic> features = await extractAudioFeatures(tempFile.path);

      // Tambahkan ke audioStore dengan label yang sesuai
      audioStore.add({
        'label': 0,
        'features': features,
      });
    }

    // Setelah semua fitur dari assets ditambahkan, simpan ke file XML
    await saveFeaturesToXml(audioStore, filePath);
    print('All audio features from assets have been saved to XML file.');
  }

  Future<void> saveSVMParametersToXml(List<double> weights, double bias) async {
    final builder = xml.XmlBuilder();
    builder.processing('xml', 'version="1.0"');
    builder.element('SVMParameters', nest: () {
      builder.element('Weights', nest: () {
        for (var weight in weights) {
          builder.element('Weight', nest: weight.toString());
        }
      });
      builder.element('Bias', nest: bias.toString());
    });

    final directory = await getExternalStorageDirectory();
    final filePath = '${directory!.path}/svm_parameters.xml';
    final file = File(filePath);
    file.writeAsStringSync(builder.buildDocument().toString());

    print('Saved best SVM parameters to: $filePath');
  }

  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // =================== SVM ========================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  // Fungsi untuk melatih SVM dengan hyperparameter yang diberikan
  void _trainSVM(List<Map<String, dynamic>> trainingData, double learningRate, int numEpochs, double C) {
    int fixedMfccLength = 40;
    int numFeatures = fixedMfccLength + 2;
    svmWeights = List<double>.filled(numFeatures, 0.0);
    svmBias = 0.0;

    for (int epoch = 0; epoch < numEpochs; epoch++) {
      for (var sample in trainingData) {
        List<double> features = [
          sample['features']['amplitude'],
          sample['features']['decibel']
        ];

        List<double> mfccs = sample['features']['mfcc'];
        if (mfccs.length > fixedMfccLength) {
          mfccs = mfccs.sublist(0, fixedMfccLength);
        } else if (mfccs.length < fixedMfccLength) {
          mfccs = List<double>.from(mfccs)..addAll(List<double>.filled(fixedMfccLength - mfccs.length, 0.0));
        }
        features.addAll(mfccs);
        int label = sample['label'];
        double prediction = _svmPredict(features);
        double margin = label * prediction;

        if (margin < 1) {
          double slack = 1 - margin;
          for (int i = 0; i < svmWeights.length; i++) {
            svmWeights[i] += learningRate * (label * features[i] - 2 * 0.01 * svmWeights[i]);
          }
          svmBias += learningRate * (label - C * slack);
        } else {
          for (int i = 0; i < svmWeights.length; i++) {
            svmWeights[i] += learningRate * (-2 * 0.01 * svmWeights[i]);
          }
        }
      }
    }
  }

  double _svmPredict(List<double> features) {
    int minLength = min(svmWeights.length, features.length);
    List<double> trimmedFeatures = features.sublist(0, minLength);
    List<double> trimmedWeights = svmWeights.sublist(0, minLength);

    double result = svmBias;
    for (int i = 0; i < trimmedWeights.length; i++) {
      result += trimmedWeights[i] * trimmedFeatures[i];
    }
    return result;
  }

  double evaluateSVM(List<Map<String, dynamic>> validationData) {
    int correctPredictions = 0;
    for (var sample in validationData) {
      List<double> features = [
        sample['features']['amplitude'],
        sample['features']['decibel'],
        ...sample['features']['mfcc']
      ];
      int label = sample['label'];
      double prediction = _svmPredict(features);
      if ((prediction > 0 && label == 1) || (prediction <= 0 && label == -1)) {
        correctPredictions++;
      }
    }
    return correctPredictions / validationData.length;
  }

  void tuneAndTrainSVM(List<Map<String, dynamic>> trainingData, List<Map<String, dynamic>> validationData) {
    List<double> learningRates = [0.001, 0.01, 0.1];
    List<int> epochOptions = [50, 100, 150];
    List<double> penaltyValues = [0.1, 1.0, 10.0];

    double bestAccuracy = 0.0;

    for (double learningRate in learningRates) {
      for (int numEpochs in epochOptions) {
        for (double C in penaltyValues) {
          _trainSVM(trainingData, learningRate, numEpochs, C);
          double accuracy = evaluateSVM(validationData);

          if (accuracy > bestAccuracy) {
            bestAccuracy = accuracy;
            bestLearningRate = learningRate;
            bestEpoch = numEpochs;
            bestC = C;
            bestWeights = List.from(svmWeights);
            bestBias = svmBias;
          }
        }
      }
    }

    print('Best Parameters - Learning Rate: $bestLearningRate, Epochs: $bestEpoch, C: $bestC');
    print('Best Accuracy: $bestAccuracy');
    
    // Simpan parameter terbaik ke dalam file XML
    saveSVMParametersToXml(bestWeights, bestBias);
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // =============== NEURAL NETWORK =================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  // Fungsi untuk mengambil MFCC dengan panjang tetap
  List<double> _getFixedLengthMFCCs(List<double> mfccs, int fixedLength) {
    if (mfccs.length >= fixedLength) {
      return mfccs.sublist(0, fixedLength); // Potong MFCC jika panjangnya lebih dari fixedLength
    } else {
      // Jika panjang MFCC kurang, tambahkan 0 sampai mencapai fixedLength
      return mfccs + List<double>.filled(fixedLength - mfccs.length, 0.0);
    }
  }

  // Fungsi untuk menyimpan model ke file
  Future<void> _saveModelToFile(ANN ann, String filename) async {
    try {
      final directory = await getExternalStorageDirectory();
      final file = File('${directory?.path}/$filename');
      
      // Menggunakan method toJson() bawaan dari library
      final modelJson = ann.toJson(withIndent: true);
      
      // Simpan JSON ke file
      await file.writeAsString(modelJson);
      print('Model berhasil disimpan ke: ${file.path}');
    } catch (e) {
      print('Error saat menyimpan model: $e');
    }
  }

  // Neural network to replace SVM training
  void _trainANN(List<Map<String, dynamic>> trainingData) {
    var stopwatchTrain = Stopwatch()..start();
    int fixedMfccLength = 40;
    // int numFeatures = fixedMfccLength + 2;
    var scale = ScaleDouble.ZERO_TO_ONE;

    var samples = trainingData.map((sample) {
      List<double> features = [
        ..._getFixedLengthMFCCs(sample['features']['mfcc'], fixedMfccLength)
      ];
      int label = sample['label'];
      // print('ini label $label');
      return '${features.join(",")}=$label';
    }).toList();

    var sampleSet = SamplesSet(
      SampleFloat32x4.toListFromString(samples, scale, false),
      subject: 'audio_classification'
    );

    var activationFunction = ActivationFunctionSigmoid();
    var ann = ANN(
      scale,
      LayerFloat32x4(fixedMfccLength, true, ActivationFunctionLinear()),
      [HiddenLayerConfig(10, true, activationFunction)],
      LayerFloat32x4(1, false, activationFunction)
    );

    print('Starting ANN training...');

    var backpropagation = Backpropagation(ann, sampleSet);
    var targetError = backpropagation.trainUntilGlobalError(
      targetGlobalError: 0.01,
      maxEpochs: 800,
      maxRetries: 10
    );

    print('Training complete. Achieved target error: $targetError');
    stopwatchTrain.stop();

    print("Prediction time Train: ${stopwatchTrain.elapsedMilliseconds} ms");

    // Menghitung global error dan detail prediksi
    var globalError = ann.computeSamplesGlobalError(sampleSet.samples);
    print('\nTraining Evaluation:');
    print('Global Error: $globalError');
    
    int correctPredictions = 0;
    print('\nSamples Outputs:');
    for (var i = 0; i < sampleSet.samples.length; ++i) {
      var sample = sampleSet.samples[i];
      var input = sample.input;
      var expected = sample.output;
      
      // Activate the sample input
      ann.activate(input);
      var output = ann.output;
      
      // Convert output to binary prediction (0 or 1)
      var predictedLabel = output[0] >= 0.5 ? 1.0 : 0.0;
      var actualLabel = expected[0];
      
      // Check if prediction is correct
      if ((predictedLabel - actualLabel).abs() < 0.1) { // Using threshold for floating point comparison
        correctPredictions++;
      }
      
      print('Sample $i:');
      print('  Input: $input');
      print('  Predicted: $output (Binary: $predictedLabel)');
      print('  Expected: $expected');
      print('  Error: ${output[0] - expected[0]}');
    }

    // Calculate and print accuracy
    double accuracy = correctPredictions / sampleSet.samples.length;
    print('\nTraining Metrics:');
    print('Accuracy: ${(accuracy * 100).toStringAsFixed(2)}%');
    print('Correct Predictions: $correctPredictions/${sampleSet.samples.length}');

    // Simpan model ke file menggunakan method bawaan
    _saveModelToFile(ann, 'voice_recognition_model.json');
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // =================== RECORD =====================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<void> _startRecording() async {
    try {
      if (recordingCount >= 6) {
        print('maximum recording reached.');
        return;
      }

      final directory = await getExternalStorageDirectory();
      _filePath = '${directory?.path}/audio_record.wav';

      var status = await Permission.microphone.request();
      if (!status.isGranted) {
        print("Microphone permission is not granted");
        return;
      }

      // Start recorder with level monitoring
      await _recorder.startRecorder(
        toFile: _filePath,
        codec: Codec.pcm16WAV,
        sampleRate: sampleRate
      );

      // Setup amplitude monitoring
      _recorderSubscription = _recorder.onProgress!.listen((event) {
        if (event.decibels != null) {
          setState(() {
            // Konversi dB ke amplitude
            _amplitude = pow(10, (event.decibels! / 20)) as double;
            // _decibel = event.decibels!;
            
            // Update flags berdasarkan threshold
            _isAmplitudeHigh = _amplitude > 1200;
            _isAmplitudeLow = _amplitude >= 50 && _amplitude <= 600;
            
            // Debug print
            if (_isAmplitudeHigh) {
              print('High amplitude detected: $_amplitude');
            } else if (_isAmplitudeLow) {
              print('Low amplitude detected: $_amplitude');
            }
          });
        }
      });

      setState(() {
        _isRecording = true;
      });

      // Stop recording after 2 seconds
      Timer(Duration(seconds: 2), () async {
        await _stopRecording();
      });
    } catch (e) {
      print('Error starting recording: $e');
    }
  }

  Future<List<int>> _getAudioContent(String path) async {
    return File(path).readAsBytesSync().toList();
  }

  // Modifikasi pada fungsi _stopRecording
  Future<void> _stopRecording() async {
    try {
      // Cancel amplitude monitoring subscription
      await _recorderSubscription?.cancel();
      _recorderSubscription = null;

      if (_recorder.isRecording) {
        await _recorder.stopRecorder();
        print('Recording stopped.');
      }

      setState(() {
        _isRecording = false;
        showError = false;
      });

      final audio = await _getAudioContent(_filePath!);
      final response = await speechToText.recognize(config, audio);
      String? detectedText = response.results
          .map((result) => result.alternatives.first.transcript)
          .join(' ');

      // Kondisi untuk menyimpan rekaman berdasarkan amplitudo
      bool shouldSaveRecording = false;
      
      print('Amplitude: $_amplitude');

      // Amplitudo rendah pada indeks rekaman genap
      if (_isAmplitudeLow && recordingCount % 2 == 0) {
        shouldSaveRecording = true;
        print('Saving low amplitude recording');
      }
      // Amplitudo tinggi pada indeks rekaman ganjil
      if (_isAmplitudeHigh && recordingCount % 2 != 0) {
        shouldSaveRecording = true;
        print('Saving high amplitude recording');
      }

      if (detectedText == keywords[recordingCount] && shouldSaveRecording) {
        setState(() {
          recordingCount++;
        });
        print("Correct keyword detected: $detectedText");

        List<Map<String, dynamic>> audioStore = [];

        if (_filePath != null) {
          Map<String, dynamic> recordedFeatures = await extractAudioFeatures(_filePath!);
          audioStore.add({
            'label': 1,
            'features': recordedFeatures,
            // 'amplitude_flutter_sound': _amplitude,  // Menyimpan nilai amplitude
            // 'decibel_flutter_sound': _decibel,
            // 'isHigh': _isAmplitudeHigh,
            // 'isLow': _isAmplitudeLow
          });
        }
        var stopwatchEkstrakRecord = Stopwatch()..start();
        // Save features to XML
        final directory = await getExternalStorageDirectory();
        if (directory != null) {
          String filePath = '${directory.path}/audio_features.xml';
          await saveFeaturesToXml(audioStore, filePath);
          print('Audio features saved successfully.');
        }
        stopwatchEkstrakRecord.stop();
        print("Prediction time Ekstrak Record: ${stopwatchEkstrakRecord.elapsedMilliseconds} ms");

        if (recordingCount == 6) {
          var stopwatchAudio = Stopwatch()..start();
          final directory = await getExternalStorageDirectory();
          if (directory != null) {
            String filePath = '${directory.path}/audio_features.xml';
            await loadAndAppendFeaturesFromAssets([], filePath);
            print('Audio features from assets saved successfully.');
            stopwatchAudio.stop();
            print("Prediction time Ekstrak Other Audio: ${stopwatchAudio.elapsedMilliseconds} ms");

            await loadFeaturesFromXml();
            _trainANN(trainingData);
            print('Model training completed.');
          }
        }
      } else {
        setState(() {
          showError = true;
        });
        print("Incorrect keyword or amplitude. Expected: ${keywords[recordingCount]}, Detected: $detectedText");
        print("Amplitude: $_amplitude, High: $_isAmplitudeHigh, Low: $_isAmplitudeLow");
      }
    } catch (e) {
      print('Error stopping recorder: $e');
    }
  }

  Future<void> _resetRecording() async {
    try {
      // Get the directory and XML file path
      final directory = await getExternalStorageDirectory();
      if (directory != null) {
        final filePathFeature = '${directory.path}/audio_features.xml';
        final filefeature = File(filePathFeature);
        final filePathSvm = '${directory.path}/svm_parameters.xml';
        final filesvm = File(filePathSvm);
        final filePathAnn = '${directory.path}/voice_recognition_model.json';
        final fileAnn = File(filePathAnn);

        // Delete the XML file if it exists
        if (await filefeature.exists()) {
          await filefeature.delete();
          print('XML Feature file deleted.');
        }
        if (await filesvm.exists()) {
          await filesvm.delete();
          print('XML SVM file deleted.');
        }
        if (await fileAnn.exists()) {
          await fileAnn.delete();
          print('JSON ANN file deleted.');
        }
      }

      // Reset recording count and error state
      setState(() {
        recordingCount = 0;
        showError = false;
      });

      print('Recording count reset.');
    } catch (e) {
      print('Error resetting recording: $e');
    }
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


  @override
  Widget build(BuildContext context) {
    // Helper function untuk mendapatkan instruksi volume
    String getVolumeInstruction() {
      if (recordingCount >= keywords.length) return '';
      return recordingCount % 2 == 0 
          ? '(Berbisik)' 
          : '(Berteriak)';
    }

    // Helper function untuk mendapatkan warna instruksi
    Color getInstructionColor() {
      if (recordingCount >= keywords.length) return Colors.green;
      return recordingCount % 2 == 0 
          ? Colors.blue 
          : Colors.red;
    }

    // Helper function untuk mendapatkan icon instruksi
    IconData getVolumeIcon() {
      if (recordingCount >= keywords.length) return Icons.check_circle;
      return recordingCount % 2 == 0 
          ? Icons.volume_down 
          : Icons.volume_up;
    }

    return Scaffold(
      appBar: AppBar(
        title: Text('Save Features', style: TextStyle(fontSize: 24, color: Colors.white)),
        centerTitle: true,
        backgroundColor: Colors.deepPurple,
      ),
      body: Padding(
        padding: EdgeInsets.all(20.0),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: <Widget>[
              // Title and Instructions
              Card(
                color: Colors.purple[50],
                elevation: 4,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Padding(
                  padding: EdgeInsets.all(16.0),
                  child: recordingCount < keywords.length
                      ? Column(
                          children: [
                            Text(
                              'Instruksi Perekaman',
                              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.deepPurple),
                            ),
                            SizedBox(height: 10),
                            Row(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(
                                  getVolumeIcon(),
                                  color: getInstructionColor(),
                                  size: 24,
                                ),
                                SizedBox(width: 8),
                                Text(
                                  getVolumeInstruction(),
                                  style: TextStyle(
                                    fontSize: 18,
                                    fontWeight: FontWeight.bold,
                                    color: getInstructionColor(),
                                  ),
                                ),
                              ],
                            ),
                            SizedBox(height: 10),
                            Text(
                              'Ucapkan Kata: "${keywords[recordingCount]}"',
                              style: TextStyle(fontSize: 18, color: Colors.deepPurple[700]),
                            ),
                          ],
                        )
                      : Column(
                          children: [
                            Text(
                              'Record sudah sesuai',
                              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.green[700]),
                            ),
                            Icon(Icons.check_circle, color: Colors.green, size: 40),
                          ],
                        ),
                ),
              ),
              SizedBox(height: 20),

              // Recording Progress Indicator
              Container(
                padding: EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                decoration: BoxDecoration(
                  color: Colors.purple[100],
                  borderRadius: BorderRadius.circular(15),
                ),
                child: Column(
                  children: [
                    Text(
                      'Progress Rekaman',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        color: Colors.deepPurple,
                      ),
                    ),
                    SizedBox(height: 8),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: List.generate(6, (index) {
                        return Container(
                          margin: EdgeInsets.symmetric(horizontal: 4),
                          width: 20,
                          height: 20,
                          decoration: BoxDecoration(
                            shape: BoxShape.circle,
                            color: index < recordingCount 
                                ? Colors.green 
                                : index == recordingCount 
                                    ? (index % 2 == 0 ? Colors.blue : Colors.red)
                                    : Colors.grey[300],
                          ),
                          child: index < recordingCount
                              ? Icon(Icons.check, size: 14, color: Colors.white)
                              : null,
                        );
                      }),
                    ),
                    SizedBox(height: 8),
                    Text(
                      '${recordingCount}/6 rekaman',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.deepPurple[700],
                      ),
                    ),
                  ],
                ),
              ),
              SizedBox(height: 20),

              // Microphone Button
              GestureDetector(
                onTap: _isRecording || recordingCount >= keywords.length ? null : _startRecording,
                child: Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    color: _isRecording 
                        ? Colors.red 
                        : recordingCount >= keywords.length 
                            ? Colors.grey 
                            : Colors.deepPurple,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(color: Colors.black26, blurRadius: 8, offset: Offset(0, 4)),
                    ],
                  ),
                  child: Icon(
                    _isRecording ? Icons.mic : Icons.mic_off,
                    color: Colors.white,
                    size: 36,
                  ),
                ),
              ),
              SizedBox(height: 15),

              // Error Message
              if (showError)
                Container(
                  padding: EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                  decoration: BoxDecoration(
                    color: Colors.red[100],
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.error_outline, color: Colors.red),
                      SizedBox(width: 8),
                      Text(
                        'Kata tidak sesuai, coba lagi.',
                        style: TextStyle(color: Colors.red, fontSize: 16),
                      ),
                    ],
                  ),
                ),
              SizedBox(height: 20),

              // Navigation Buttons
              Wrap(
                spacing: 10,
                runSpacing: 10,
                alignment: WrapAlignment.center,
                children: [
                  ElevatedButton.icon(
                    onPressed: _resetRecording,
                    icon: Icon(Icons.refresh),
                    label: Text("Reset"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.red,
                      foregroundColor: Colors.white,
                      padding: EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                    ),
                  ),
                  ElevatedButton.icon(
                    onPressed: () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => SoundAnalyzer()),
                    ),
                    icon: Icon(Icons.analytics),
                    label: Text("KNN"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.blue,
                      foregroundColor: Colors.white,
                    ),
                  ),
                  ElevatedButton.icon(
                    onPressed: () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => SoundAnalyzerSvm()),
                    ),
                    icon: Icon(Icons.analytics_outlined),
                    label: Text("SVM"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.green,
                      foregroundColor: Colors.white,
                    ),
                  ),
                  ElevatedButton.icon(
                    onPressed: () => Navigator.push(
                      context,
                      MaterialPageRoute(builder: (context) => SoundAnalyzerAnn()),
                    ),
                    icon: Icon(Icons.architecture),
                    label: Text("ANN"),
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.orange,
                      foregroundColor: Colors.white,
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}