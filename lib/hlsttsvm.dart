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
import 'package:google_speech/google_speech.dart';
import 'package:ffmpeg_kit_flutter/ffmpeg_kit.dart';
import 'package:fftea/fftea.dart';
import 'package:xml/xml.dart' as xml;

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
      home: SoundAnalyzerSvm(),
    );
  }
}

class SoundAnalyzerSvm extends StatefulWidget {
  @override
  _SoundAnalyzerState createState() => _SoundAnalyzerState();
}

class _SoundAnalyzerState extends State<SoundAnalyzerSvm> {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  final speechToText = SpeechToText.viaApiKey("AIzaSyDx5fZpE0z1QxYV7mwN1cvxig7tUvzw4Xc");
  final config = RecognitionConfig(
    encoding: AudioEncoding.LINEAR16,
    model: RecognitionModel.basic,
    enableAutomaticPunctuation: true,
    sampleRateHertz: 16000,
    languageCode: 'id-ID',
  );
  bool _isRecording = false;
  double _frequency = 0.0;
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

    // if (normalizedData.every((sample) => sample == 0)) {
    //   throw Exception("Audio normalization failed. All samples are zero.");
    // }

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
    var audioSignal = normalizeAudioData(audioBytes);

    final chunkSize = 512;
    final stft = STFT(chunkSize, Window.hanning(chunkSize));
    final spectrogram = <Float64List>[];

    stft.run(audioSignal, (Float64x2List freq) {
      final magnitudes = freq.discardConjugates().magnitudes();
      spectrogram.add(magnitudes);
    });

    var melFilters = melFilterbank(26, chunkSize, sampleRate);
    var melSpectrogram = <List<double>>[];
    for (var frame in spectrogram) {
      var melEnergies = applyMelFilterbank(frame, melFilters);
      melSpectrogram.add(melEnergies);
    }

    for (var i = 0; i < melSpectrogram.length; i++) {
      for (var j = 0; j < melSpectrogram[i].length; j++) {
        melSpectrogram[i][j] = log(melSpectrogram[i][j] + 1e-10);
      }
    }

    var mfccList = <double>[];
    for (var frame in melSpectrogram) {
      var dctCoeffs = dct(frame, numCoefficients);
      mfccList.addAll(dctCoeffs);
    }

    return mfccList;
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

    // FFT to calculate dominant frequency
    final fft = FFT(audioList.length);
    final freqs = fft.realFft(audioList);

    // Spectral magnitude
    List<double> freqsDouble = freqs.map((f) => sqrt(f.x * f.x + f.y * f.y)).toList();
    double maxAmplitude = freqsDouble.reduce((curr, next) => curr.abs() > next.abs() ? curr : next);
    int maxFreqIndex = freqsDouble.indexOf(maxAmplitude);

    // Dominant frequency calculation
    double dominantFreq = (maxFreqIndex * 16000) / (audioList.length); // Divide by 2 for FFT symmetry

    // Decibel calculation
    double minAmplitude = 1e-10;
    maxAmplitude = maxAmplitude.abs();
    if (maxAmplitude < minAmplitude) {
      maxAmplitude = minAmplitude;
    }
    double decibel = 20 * log(maxAmplitude) / log(10);

    // MFCC Calculation (13 coefficients as an example)
    String pcmPath = await _extractWaveform(filePath);
    final audioBytes = await _getAudioBytes(pcmPath);
    List<double> sampleMFCC = computeMFCC(audioBytes, 16000, 13);

    // Return the extracted features as a map
    return {
      'frequency': dominantFreq,
      'amplitude': maxAmplitude,
      'decibel': decibel,
      'mfcc': sampleMFCC,
    };
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // ==================LOAD XML =====================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<void> loadSVMParametersFromXml() async {
    final directory = await getExternalStorageDirectory();
    if (directory != null) {
      String filePath = '${directory.path}/svm_parameters.xml';
      final file = File(filePath);

      if (!file.existsSync()) {
        print('File tidak ditemukan: $filePath');
        return;
      }

      // Baca konten file XML
      final xmlString = await file.readAsString();
      final xmlDoc = xml.XmlDocument.parse(xmlString);

      // Ambil nilai bias
      final biasElement = xmlDoc.findAllElements('Bias').first;
      svmBias = double.parse(biasElement.text);

      // Ambil nilai weights
      final weightElements = xmlDoc.findAllElements('Weight');
      svmWeights = weightElements.map((element) => double.parse(element.text)).toList();

      print('SVM parameters loaded successfully');
      print('Bias: $svmBias');
      print('Weights: $svmWeights');
    }
  }

  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // =========== MATCHING VOICE & STT ===============
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  double _svmPredict(List<double> features) {
    int minLength = min(svmWeights.length, features.length);

    // Potong kedua vektor sesuai panjang minimum
    List<double> trimmedFeatures = features.sublist(0, minLength);
    List<double> trimmedWeights = svmWeights.sublist(0, minLength);

    double result = svmBias;
    for (int i = 0; i < trimmedWeights.length; i++) {
      result += trimmedWeights[i] * trimmedFeatures[i];
    }
    return result;
  }

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
        
        final audio = await _getAudioContent(newPath);
        final response = await speechToText.recognize(config, audio);
        String? detectedText = response.results
            .map((result) => result.alternatives.first.transcript)
            .join(' ');

        print('Kata yang terdeteksi: $detectedText');

        if (detectedText.contains('tolong') || detectedText.contains('help')) {
          print('Terdeteksi kata: $detectedText');

          await loadSVMParametersFromXml();

          List<double> newFeatures = [
            // _copy_features['frequency'] ?? 0.0,
            _copy_features['amplitude'] ?? 0.0,
            _copy_features['decibel'] ?? 0.0,
            ...?_copy_features['mfcc']
          ];

          if (newFeatures.isEmpty) {
            print("Error: newFeatures vector is empty.");
            return;
          }

          double prediction = _svmPredict(newFeatures);
          bool predictionMatch = prediction > 0;

          print('Hasil matching : $predictionMatch');

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
          print('Kata "tolong" atau "help" tidak ditemukan.');
        }
      } else {
        print('File sumber tidak ditemukan.');
      }
    } catch (e) {
      print('Error saat memproses dan membandingkan file audio: $e');
    }
  }
  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // ================= REALTIME =====================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<void> _startRecording() async {
    final directory = await getExternalStorageDirectory();
    _filePath = '${directory?.path}/audio_streaming.wav';
    // print('Audio saved to: $_filePath');

    try {
      await _recorder.startRecorder(
        toFile: _filePath,
        codec: Codec.pcm16WAV,
      );
    } catch (e) {
      print('Error in starting recorder: $e');
    }

    _recorder.onProgress!.listen((event) async {
      if (_filePath != null) {
        // Extract features
        Map<String, dynamic> realtime_features = await extractAudioFeatures(_filePath!);

        setState(() {
          _amplitude = pow(10, (event.decibels! / 20)) as double;
          _decibel = event.decibels!;
          _frequency = realtime_features['frequency'];

          _isAmplitudeHigh = _amplitude > 1000;
          _isAmplitudeLow = _amplitude >= 50 && _amplitude <= 120;

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
      }
    });

    setState(() {
      _isRecording = true;
    });

    _recordingTimer = Timer.periodic(Duration(milliseconds: 2800), (timer) async {
      await _restartRecording();
    });
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
        title: Text('Real-time Sound Analysis SVM'),
      ),
      body: Container(
        color: _isAmplitudeHigh
            ? Colors.red
            : _isAmplitudeLow
                ? Colors.blue
                : Colors.white,
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            SizedBox(height: 20),
            Text(
              'Frekuensi: ${_frequency.toStringAsFixed(2)}',
              style: TextStyle(fontSize: 20),
            ),
            SizedBox(height: 20),
            Text(
              'Amplitudo: ${_amplitude.toStringAsFixed(2)}',
              style: TextStyle(fontSize: 20),
            ),
            SizedBox(height: 20),
            Text(
              'Desibel: ${_decibel.toStringAsFixed(2)} dB',
              style: TextStyle(fontSize: 20),
            ),
            SizedBox(height: 20),
            if (_isAmplitudeHigh)
              Text(
                'Peringatan: Suara terlalu tinggi!',
                style: TextStyle(fontSize: 24, color: Colors.white, fontWeight: FontWeight.bold),
              ),
            if (_isAmplitudeLow)
              Text(
                'Keterangan: Suara rendah',
                style: TextStyle(fontSize: 24, color: Colors.white, fontWeight: FontWeight.bold),
              ),
            SizedBox(height: 50),
            ElevatedButton(
              onPressed: _isRecording ? _stopRecording : _startRecording,
              child: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
            ),
            if (_comparisonResult['detectedText'] != '') 
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Text(
                  'Kata terdeteksi: ${_comparisonResult['detectedText']} \nCocok dengan User: ${_comparisonResult['isMatching'] ? "Ya" : "Tidak"}',
                  style: TextStyle(fontSize: 18, color: Colors.green),
                ),
              ),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: () {
                Navigator.pop(context);
              },
              child: Text("Back to Recorder"),
            ),
          ],
        ),
      ),
    );
  }
}