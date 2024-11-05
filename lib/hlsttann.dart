import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'dart:async';
import 'package:permission_handler/permission_handler.dart';
import 'package:ffmpeg_kit_flutter/ffmpeg_kit.dart';
import 'package:google_speech/google_speech.dart';
import 'package:fftea/fftea.dart';
import 'package:eneural_net/eneural_net.dart';
import 'package:flutter_sound_processing/flutter_sound_processing.dart';
import 'dart:core';

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
      title: 'Real-time Sound Analysis',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: SoundAnalyzerAnn(),
    );
  }
}

class SoundAnalyzerAnn extends StatefulWidget {
  @override
  _SoundAnalyzerState createState() => _SoundAnalyzerState();
}

class _SoundAnalyzerState extends State<SoundAnalyzerAnn> {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  final speechToText = SpeechToText.viaApiKey("AIzaSyBgM6CzvKTZVEit_d-kOxjfstTW4OWe6ac");
  final config = RecognitionConfig(
    encoding: AudioEncoding.LINEAR16,
    model: RecognitionModel.command_and_search,
    enableAutomaticPunctuation: true,
    sampleRateHertz: 16000,
    languageCode: 'id-ID',
  );
  bool _isRecording = false;
  // double _frequency = 0.0;
  double _amplitude = 0.0;
  double _decibel = 0.0;
  bool _isAmplitudeHigh = false;
  bool _isAmplitudeLow = false;
  Timer? _recordingTimer;
  String? _filePath;
  Map<String, dynamic> _comparisonResult = {'detectedText': '', 'isMatching': false};
  Map<String, dynamic> _copy_features = {};
  List<Map<String, dynamic>> nearestNeighbors = [];
  String majorityLabel = '';
  List<double> svmWeights = [];
  double svmBias = 0.0;
  late ANN _trainedANN;

  @override
  void initState() {
    super.initState();
    _initRecorder();
  }

  Future<void> _initRecorder() async {
    // Request permissions for microphone and storage
    var status = await Permission.microphone.request();
    if (status != PermissionStatus.granted) {
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
    try {
      // Read the file
      File file = File(filePath);
      Uint8List audioData = await file.readAsBytes();

      // Convert Uint8List to List<int> (16-bit PCM)
      List<int> audioListInt = [];
      for (int i = 0; i < audioData.length - 1; i += 2) {
        int value = (audioData[i + 1] << 8) | audioData[i];
        if (value >= 0x8000) value -= 0x10000;
        audioListInt.add(value);
      }

      // Convert to List<double> dan pastikan panjang sesuai bufferSize
      List<double> audioList = audioListInt.map((e) => e.toDouble()).toList();
      
      // Trim atau pad audio list untuk memastikan panjang tepat bufferSize
      if (audioList.length > bufferSize) {
        audioList = audioList.sublist(0, bufferSize);
      } else if (audioList.length < bufferSize) {
        audioList.addAll(List<double>.filled(bufferSize - audioList.length, 0.0));
      }

      // Remove DC component (mean removal)
      double mean = audioList.reduce((a, b) => a + b) / audioList.length;
      audioList = audioList.map((v) => v - mean).toList();

      // Normalize amplitude
      double maxAbsVal = audioList.map((v) => v.abs()).reduce((a, b) => a > b ? a : b);
      if (maxAbsVal > 0) {
        audioList = audioList.map((v) => v / maxAbsVal).toList();
      }

      // Calculate number of frames
      int numFrames = ((audioList.length - fftSize) ~/ hopLength) + 1;
      int paddedLength = ((numFrames - 1) * hopLength) + fftSize;
      
      // Pad signal if necessary
      if (audioList.length < paddedLength) {
        audioList.addAll(List<double>.filled(paddedLength - audioList.length, 0.0));
      }

      // MFCC Calculation
      final flutterSoundProcessingPlugin = FlutterSoundProcessing();
      
      // Ensure signals length is multiple of hopLength
      int signalLength = ((audioList.length ~/ hopLength) * hopLength);
      final signals = audioList.sublist(0, signalLength);

      // Extract MFCC features dengan parameter yang sudah ditentukan
      final featureMatrix = await flutterSoundProcessingPlugin.getFeatureMatrix(
        signals: signals,
        fftSize: fftSize,
        hopLength: hopLength,
        nMels: nMels,
        mfcc: mfcc,
        sampleRate: sampleRate,
      );

      // Initialize average MFCC dengan jumlah koefisien yang sesuai
      List<double> averageMFCC = List.filled(mfcc, 0.0);

      if (featureMatrix != null && featureMatrix.isNotEmpty) {
        int frameCount = featureMatrix.length ~/ mfcc;
        
        if (frameCount > 0) {
          for (int frameIndex = 0; frameIndex < frameCount; frameIndex++) {
            for (int i = 0; i < mfcc; i++) {
              if ((frameIndex * mfcc + i) < featureMatrix.length) {
                averageMFCC[i] += featureMatrix[frameIndex * mfcc + i];
              }
            }
          }
          
          // Calculate average
          averageMFCC = averageMFCC.map((mfcc) => mfcc / frameCount).toList();
        }
      }

      return {
        'mfcc': averageMFCC,
      };
    } catch (e, stackTrace) {
      print('Error in extractAudioFeatures: $e');
      print('Stack trace: $stackTrace');
      return {
        'mfcc': List.filled(mfcc, 0.0),
      };
    }
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // ==================LOAD XML =====================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  // Fungsi untuk memuat model dari file
  Future<ANN?> _loadModelFromFile(String filename) async {
    try {
      final directory = await getExternalStorageDirectory();
      final file = File('${directory?.path}/$filename');
      
      if (!await file.exists()) {
        print('File model tidak ditemukan');
        return null;
      }

      // Baca JSON dari file
      final jsonString = await file.readAsString();
      
      // Menggunakan method fromJson() bawaan dari library
      var ann = ANN.fromJson(jsonString);
      
      print('Model berhasil dimuat dari: ${file.path}');
      return ann;
    } catch (e) {
      print('Error saat memuat model: $e');
      return null;
    }
  }

  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // =========== MATCHING VOICE & STT ===============
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<List<int>> _getAudioContent(String path) async {
    return File(path).readAsBytesSync().toList();
  }

  // Helper function to amplify audio data
  Future<void> amplifyAudioFile(File source, File destination) async {
    // Load audio data from source as bytes
    List<int> audioData = await source.readAsBytes();
    ByteData byteData = ByteData.sublistView(Uint8List.fromList(audioData));

    // Initialize list to hold amplified PCM data
    List<int> amplifiedData = [];
    const double gainFactor = 3.0;

    // Amplify each 16-bit sample (assumes input WAV is 16-bit PCM)
    for (int i = 0; i < byteData.lengthInBytes; i += 2) {
      // Read sample as signed 16-bit integer (PCM encoding)
      int sample = byteData.getInt16(i, Endian.little);

      // Amplify sample and clamp to prevent overflow
      int amplifiedSample = (sample * gainFactor).toInt();
      amplifiedSample = amplifiedSample.clamp(-32768, 32767);

      // Store amplified sample back as bytes
      amplifiedData.addAll([
        amplifiedSample & 0xFF,          // Lower byte
        (amplifiedSample >> 8) & 0xFF,   // Upper byte
      ]);
    }

    // Write amplified data to destination
    await destination.writeAsBytes(amplifiedData);
  }

  List<double> _getFixedLengthMFCCs(List<double> mfccs, int fixedLength) {
    if (mfccs.length >= fixedLength) {
      return mfccs.sublist(0, fixedLength); // Potong MFCC jika panjangnya lebih dari fixedLength
    } else {
      // Jika panjang MFCC kurang, tambahkan 0 sampai mencapai fixedLength
      return mfccs + List<double>.filled(fixedLength - mfccs.length, 0.0);
    }
  }

  Future<void> processAndCompareAudio() async {
    if (_filePath == null) {
      print('File path tidak ditemukan.');
      return;
    }

    try {
      final directory = await getExternalStorageDirectory();
      final newPath = path.join(directory!.path, 'processed_audio.wav');
      final File sourceFile = File(_filePath!);
      final File destinationFile = File(newPath);

      if (await sourceFile.exists()) {
        if (_isAmplitudeLow) {
          print('Amplitude rendah terdeteksi, memperbesar suara...');
          await amplifyAudioFile(sourceFile, destinationFile);
        } else {
          print('Amplitude tinggi terdeteksi');
          await sourceFile.copy(destinationFile.path);
        }

        print('File audio berhasil disalin ke: $newPath');
        
        var stopwatchSTT = Stopwatch()..start();
        final audio = await _getAudioContent(newPath);
        final response = await speechToText.recognize(config, audio);
        String? detectedText = response.results
            .map((result) => result.alternatives.first.transcript)
            .join(' ');

        print('Kata yang terdeteksi: $detectedText');
        stopwatchSTT.stop();
        print("Prediction time STT: ${stopwatchSTT.elapsedMilliseconds} ms");

        if (detectedText.contains('tolong') || detectedText.contains('help') || detectedText.contains('aw') || detectedText.contains('aduh')) {
          print('Terdeteksi kata: $detectedText');
          var stopwatchNN = Stopwatch()..start();

          _trainedANN = (await _loadModelFromFile('voice_recognition_model.json'))!;
          int fixedMfccLength = 40;
          // int numFeatures = fixedMfccLength + 2;

          List<double> newFeatures = [
            // _copy_features['frequency'] ?? 0.0,
            // _copy_features['amplitude'] ?? 0.0,
            // _copy_features['decibel'] ?? 0.0,
            ..._getFixedLengthMFCCs(_copy_features['mfcc'], fixedMfccLength)
          ];

          int label = 1;

          // Prepare the data for the neural network
          var scale = ScaleDouble.ZERO_TO_ONE;
          var sample = ['${newFeatures.join(",")}=$label']; // Convert features to string format
          var sampleSet = SampleFloat32x4.toListFromString(sample, scale, false);
          var output = [];

          print("ini sample untuk predict: $sampleSet");
          for (var sample in sampleSet) {
            var input = sample.input; // Extract input from SampleFloat32x4
            print('input: $input');

            // Pass input to the neural network for activation
            _trainedANN.activate(input);
            output = _trainedANN.output;
            print("Prediction output: $output");
          }

          // Convert output to binary prediction (0 or 1)
          var predictedLabel = output[0] >= 0.5 ? 1.0 : 0.0;
          bool predictionMatch = false;
          // Check if prediction is correct
          if ((output[0] - label).abs() < 0.15 && predictedLabel == label) { // Using threshold for floating point comparison
            predictionMatch = true;
          }
          stopwatchNN.stop();
          print("Prediction time NN: ${stopwatchNN.elapsedMilliseconds} ms");

          setState(() {
            _comparisonResult['detectedText'] = detectedText;
            _comparisonResult['isMatching'] = predictionMatch;
          });

          // Reset setelah menampilkan hasil
          Future.delayed(Duration(seconds: 1), () {
            setState(() {
              _comparisonResult = {'detectedText': '', 'isMatching': false};

            });
          });
        } else {
          print('Kata "tolong", "help", "aw", "aduh" tidak ditemukan.');
        }
      } else {
        print('File sumber tidak ditemukan.');
      }
    } catch (e) {
      print('Error saat memproses dan membandingkan file audio: $e');
    } finally {
      // Cleanup temp file
      try {
        final directory = await getExternalStorageDirectory();
        final newPath = path.join(directory!.path, 'processed_audio.wav');
        final tempFile = File(newPath);
        if (await tempFile.exists()) {
          await tempFile.delete();
        }
      } catch (e) {
        print('Error cleaning up temp file: $e');
      }
    }
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // ================= REALTIME =====================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<void> _startRecording() async {
    final directory = await getExternalStorageDirectory();
    _filePath = '${directory?.path}/audio_streaming.wav';

    try {
      await _recorder.startRecorder(
        toFile: _filePath,
        codec: Codec.pcm16WAV,
        sampleRate: sampleRate,
      );

      _recorder.onProgress!.listen((event) async {
        if (_filePath != null ) {
          try {
            var stopwatchHL = Stopwatch()..start();
            Map<String, dynamic> realtime_features = await extractAudioFeatures(_filePath!);
            setState(() {
              _amplitude = pow(10, (event.decibels! / 20)) as double;
              _decibel = event.decibels!;

              _isAmplitudeHigh = _amplitude > 1200;
              _isAmplitudeLow = _amplitude >= 50 && _amplitude <= 600;

              if (_isAmplitudeHigh || _isAmplitudeLow) {
                if (_isAmplitudeHigh) {
                  print('Amplitude tinggi terdeteksi.');
                }

                if (_isAmplitudeLow) {
                  print('Amplitude rendah terdeteksi.');
                }
                _copy_features = realtime_features;
                processAndCompareAudio();
              }
            });
            stopwatchHL.stop();
            print("Prediction time HL: ${stopwatchHL.elapsedMilliseconds} ms");
          } catch (e) {
            print('Error extracting audio features: $e');
          }
        }
      });
      setState(() {
        _isRecording = true;
      });
      _recordingTimer = Timer.periodic(Duration(milliseconds: 10000), (timer) async {
        await _restartRecording();
      });
    } catch (e) {
      print('Error in starting recorder: $e');
    }
  }

  Future<void> _restartRecording() async {
    await _recorder.stopRecorder();

    if (_filePath != null) {
      final file = File(_filePath!);
      if (await file.exists()) {
        await file.delete();
        // print('Deleted old recording: $_filePath');
      }
    }

    final directory = await getExternalStorageDirectory();
    _filePath = '${directory?.path}/audio_streaming.wav';
    // print('Restarting recording, audio saved to: $_filePath');

    await _recorder.startRecorder(
      toFile: _filePath,
      codec: Codec.pcm16WAV,
    );
  }

  Future<void> _stopRecording() async {
    // Cancel the recording timer
    _recordingTimer?.cancel();

    // Stop the recorder and ensure it has finished stopping before updating the state
    try {
      if (_recorder.isRecording) {
        await _recorder.stopRecorder();  // Await for the recorder to stop
        print('Recording stopped.');
      }
    } catch (e) {
      print('Error stopping recorder: $e');
    }

    // Update the state after the recorder is stopped
    setState(() {
      _isRecording = false;
      _isAmplitudeHigh = false;
      _isAmplitudeLow = false;
    });
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  @override
  void dispose() {
    _recorder.closeRecorder();
    _recordingTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Real-time Sound Analysis ANN', style: TextStyle(color: Colors.white)),
        centerTitle: true,
        backgroundColor: Colors.deepPurple,
      ),
      body: Container(
        color: _isAmplitudeHigh
            ? Colors.red[100]
            : _isAmplitudeLow
                ? Colors.blue[100]
                : Colors.grey[100],
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            SizedBox(height: 20),
            // Amplitude Display
            Card(
              color: Colors.deepPurple[50],
              elevation: 4,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Icon(Icons.graphic_eq, color: Colors.deepPurple),
                    Text(
                      'Amplitudo: ${_amplitude.toStringAsFixed(2)}',
                      style: TextStyle(fontSize: 20, color: Colors.deepPurple[700]),
                    ),
                  ],
                ),
              ),
            ),
            SizedBox(height: 20),
            // Decibel Display
            Card(
              color: Colors.deepPurple[50],
              elevation: 4,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Icon(Icons.volume_up, color: Colors.deepPurple),
                    Text(
                      'Desibel: ${_decibel.toStringAsFixed(2)} dB',
                      style: TextStyle(fontSize: 20, color: Colors.deepPurple[700]),
                    ),
                  ],
                ),
              ),
            ),
            SizedBox(height: 20),
            // Warning or Information Message
            if (_isAmplitudeHigh)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8.0),
                child: Text(
                  'Peringatan: Suara terlalu tinggi!',
                  style: TextStyle(
                    fontSize: 22,
                    color: Colors.red[700],
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            if (_isAmplitudeLow)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8.0),
                child: Text(
                  'Keterangan: Suara rendah',
                  style: TextStyle(
                    fontSize: 22,
                    color: Colors.blue[700],
                    fontWeight: FontWeight.bold,
                  ),
                  textAlign: TextAlign.center,
                ),
              ),
            SizedBox(height: 20),
            // Start/Stop Recording Button
            ElevatedButton.icon(
              onPressed: _isRecording ? _stopRecording : _startRecording,
              icon: Icon(_isRecording ? Icons.stop : Icons.mic),
              label: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.deepPurple,
                foregroundColor: Colors.white,
                padding: EdgeInsets.symmetric(vertical: 16),
                textStyle: TextStyle(fontSize: 18),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
            ),
            SizedBox(height: 20),
            // Detected Word Information
            // if (_comparisonResult['detectedText'] != '')
            Card(
              color: Colors.green[50],
              elevation: 4,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(10),
              ),
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  children: [
                    Text(
                      'Kata terdeteksi: ${_comparisonResult['detectedText']}',
                      style: TextStyle(fontSize: 18, color: Colors.green[800]),
                    ),
                    Text(
                      'Cocok dengan User: ${_comparisonResult['isMatching'] ? "Ya" : "Tidak"}',
                      style: TextStyle(fontSize: 18, color: Colors.green[800]),
                    ),
                  ],
                ),
              ),
            ),
            SizedBox(height: 20),
            // Back Button
            ElevatedButton(
              onPressed: () {
                Navigator.pop(context);
              },
              child: Text("Back to Recorder"),
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.grey[600],
                foregroundColor: Colors.white,
                padding: EdgeInsets.symmetric(vertical: 14),
                textStyle: TextStyle(fontSize: 18),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(10),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}