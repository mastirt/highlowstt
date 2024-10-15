import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:path_provider/path_provider.dart';
import 'dart:async';
import 'package:permission_handler/permission_handler.dart';
import 'package:ffmpeg_kit_flutter/ffmpeg_kit.dart';
import 'package:fftea/fftea.dart';
import 'package:xml/xml.dart' as xml;
import 'hlsttknn.dart';

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

  bool _isRecording = false;
  int recordingCount = 0;
  String? _filePath;
  // Map<String, dynamic> _copy_features = {};
  List<Map<String, dynamic>> nearestNeighbors = [];
  String majorityLabel = '';

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
    double dominantFreq = (maxFreqIndex * 16000) / (audioList.length / 2); // Divide by 2 for FFT symmetry

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
  Future<List<Map<String, dynamic>>> loadFeaturesFromXml() async {
    // Load the XML file from assets
    final xmlString = await rootBundle.loadString('assets/xFeature.xml');

    // Parse the XML content
    final xmlDoc = xml.XmlDocument.parse(xmlString);

    List<Map<String, dynamic>> audioStore = [];

    // Extract data from the XML
    for (var audioElement in xmlDoc.findAllElements('Audio')) {
      String label = audioElement.findElements('Label').first.text;
      double frequency = double.parse(audioElement.findElements('Frequency').first.text);
      double amplitude = double.parse(audioElement.findElements('Amplitude').first.text);
      double decibel = double.parse(audioElement.findElements('Decibel').first.text);
      List<double> mfcc = audioElement.findElements('MFCC')
          .map((e) => double.parse(e.text))
          .toList();

      audioStore.add({
        'label': label,
        'features': {
          'frequency': frequency,
          'amplitude': amplitude,
          'decibel': decibel,
          'mfcc': mfcc,
        }
      });
    }
    return audioStore;
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

      audioElement.children.add(xml.XmlElement(xml.XmlName('Label'), [], [xml.XmlText(audioData['label'] ?? 'Unknown')]));
      audioElement.children.add(xml.XmlElement(xml.XmlName('Frequency'), [], [xml.XmlText(audioData['features']['frequency'].toString())]));
      audioElement.children.add(xml.XmlElement(xml.XmlName('Amplitude'), [], [xml.XmlText(audioData['features']['amplitude'].toString())]));
      audioElement.children.add(xml.XmlElement(xml.XmlName('Decibel'), [], [xml.XmlText(audioData['features']['decibel'].toString())]));
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
    for (int i = 1; i <= 5; i++) {
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
        'label': 'test',
        'features': features,
      });
    }

    // Setelah semua fitur dari assets ditambahkan, simpan ke file XML
    await saveFeaturesToXml(audioStore, filePath);
    print('All audio features from assets have been saved to XML file.');
  }

  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

  // ================= RECORD =====================
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  Future<void> _startRecording() async {
    try {
      if (recordingCount >= 5) {
        print('maximum recording reached.');
        return;
      }

      final directory = await getExternalStorageDirectory();
      _filePath = '${directory?.path}/audio_streaming.wav';
      print('Audio saved to: $_filePath');

      try {
        await _recorder.startRecorder(
          toFile: _filePath,
          codec: Codec.pcm16WAV,
        );
      } catch (e) {
        print('Error in starting recorder: $e');
      }

      var status = await Permission.microphone.request();
      if (!status.isGranted) {
        print("Microphone permission is not granted");
        return; // Jika izin tidak diberikan, tidak melanjutkan perekaman
      }

      setState(() {
        _isRecording = true;
      });

      Timer(Duration(seconds: 2), () async {
        await _stopRecording();
      });
    } catch (e) {
      print('Error starting recording: $e');
    }
  }

  // Modifikasi pada fungsi _stopRecording
  Future<void> _stopRecording() async {
    try {
      if (_recorder.isRecording) {
        await _recorder.stopRecorder();
        print('Recording stopped.');
      }

      setState(() {
        _isRecording = false;
        recordingCount++;
      });

      List<Map<String, dynamic>> audioStore = [];

      if (_filePath != null) {
        Map<String, dynamic> recordedFeatures = await extractAudioFeatures(_filePath!);
        audioStore.add({
          'label': 'user',
          'features': recordedFeatures,
        });
      }

      // Simpan fitur dari rekaman ke file XML
      final directory = await getExternalStorageDirectory();
      if (directory != null) {
        String filePath = '${directory.path}/audio_features.xml';
        await saveFeaturesToXml(audioStore, filePath);
        print('Audio features saved successfully.');
      }

      // Tambahkan fitur dari assets ke XML ketika recordingCount mencapai 5
      if (recordingCount == 5) {
        final directory = await getExternalStorageDirectory();
        if (directory != null) {
          String filePath = '${directory.path}/audio_features.xml';
          await loadAndAppendFeaturesFromAssets([], filePath); // Menggunakan array kosong agar tidak duplikasi
          print('Audio features from assets saved successfully.');
        }
      }
    } catch (e) {
      print('Error stopping recorder: $e');
    }
  }



  // ================================================
  // >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Save Features'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            // Widgets yang sudah ada
            GestureDetector(
              onTap: _isRecording || recordingCount >= 5 ? null : _startRecording,
              child: Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  color: _isRecording ? Colors.red : Colors.brown[400],
                  shape: BoxShape.circle,
                ),
                child: Icon(
                  _isRecording ? Icons.mic : Icons.mic_off,
                  color: Colors.white,
                  size: 40,
                ),
              ),
            ),
            SizedBox(height: 20),
            Text(
              'Recording ${recordingCount}/5',
              style: TextStyle(fontSize: 18),
            ),
            SizedBox(height: 20),
            // Tombol Navigasi
            ElevatedButton(
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => SoundAnalyzer()),
                );
              },
              child: Text("Go to Sound Analyzer"),
            ),
          ],
        ),
      ),
    );
  }
}