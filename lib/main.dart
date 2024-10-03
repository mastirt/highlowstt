import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:flutter_fft/flutter_fft.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as path;
import 'dart:async';
import 'package:permission_handler/permission_handler.dart';
import 'package:google_speech/google_speech.dart';

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
      home: SoundAnalyzer(),
    );
  }
}

class SoundAnalyzer extends StatefulWidget {
  @override
  _SoundAnalyzerState createState() => _SoundAnalyzerState();
}

class _SoundAnalyzerState extends State<SoundAnalyzer> {
  final FlutterSoundRecorder _recorder = FlutterSoundRecorder();
  final FlutterFft _flutterFft = FlutterFft();
  final speechToText = SpeechToText.viaApiKey("");
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
  StreamSubscription? _subscription;
  Timer? _recordingTimer;
  String? _filePath;
  String? _comparisonResult;

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

  Future<void> _startRecording() async {
    final directory = await getExternalStorageDirectory();
    _filePath = '${directory?.path}/audio_streaming.wav';
    print('Audio saved to: $_filePath');

    try {
      await _recorder.startRecorder(
        toFile: _filePath,
        codec: Codec.pcm16WAV,
        // sampleRate: 16000,
        // numChannels: 1,
      );
    } catch (e) {
      print('Error in starting recorder: $e');
    }

    await _flutterFft.startRecorder();

    _subscription = _flutterFft.onRecorderStateChanged.listen((data) {
      setState(() {
        _frequency = double.tryParse(data[1].toString()) ?? 0.0;
      });
    });

    _recorder.onProgress!.listen((event) {
      if (event.decibels != null) {
        setState(() {
          _amplitude = pow(10, (event.decibels! / 20)) as double;
          _decibel = event.decibels!;

          _isAmplitudeHigh = _amplitude > 1000;
          _isAmplitudeLow = _amplitude >= 50 && _amplitude <= 100;

          if (_isAmplitudeHigh || _isAmplitudeLow) {
            print('==================================');
            if (_isAmplitudeHigh) {
              print('Amplitude tinggi terdeteksi.');
            }
            
            if (_isAmplitudeLow) {
              print('Amplitude rendah terdeteksi.');
            }
            _copyAudioFile();
          }
        });
      }
    });

    setState(() {
      _isRecording = true;
    });

    _recordingTimer = Timer.periodic(Duration(milliseconds: 2500), (timer) async {
      await _restartRecording();
    });
  }

  Future<List<int>> _getAudioContent(String path) async {
    return File(path).readAsBytesSync().toList();
  }

  Future<void> _copyAudioFile() async {
    if (_filePath != null) {
      try {
        final directory = await getExternalStorageDirectory();
        final newPath = path.join(directory!.path, 'processed_audio.wav');
        final File sourceFile = File(_filePath!);
        final File destinationFile = File(newPath);
        print('Source Path: ${sourceFile}');
        print('File Path: ${_filePath}');
        print('Directory Path: ${directory}');
        print('New Path: ${newPath}');

        if (await sourceFile.exists()) {
          await sourceFile.copy(destinationFile.path);
          print('File audio berhasil disalin ke: $newPath');
          final audio = await _getAudioContent(newPath);
          
          final response = await speechToText.recognize(config, audio);
          String? detectedText = response.results
              .map((result) => result.alternatives.first.transcript)
              .join(' ');
          
          print('Kata yang terdeteksi ${detectedText}');

          if (detectedText.contains('tolong') || detectedText.contains('help')) {
            print('Terdeteksi kata: $detectedText');
            setState(() {
              _comparisonResult = "Terdeteksi kata: $detectedText";
            });
          } else {
            print('Kata "tolong" atau "help" tidak ditemukan. Menghapus file audio.');
            await destinationFile.delete();
          }
        } else {
          print('File sumber tidak ditemukan.');
        }
      } catch (e) {
        print('Error saat menyalin file audio: $e');
      }
    }
  }

  Future<void> _restartRecording() async {
    await _recorder.stopRecorder();
    await _flutterFft.stopRecorder();
    _subscription?.cancel();

    if (_filePath != null) {
      final file = File(_filePath!);
      if (await file.exists()) {
        await file.delete();
        print('Deleted old recording: $_filePath');
      }
    }

    final directory = await getExternalStorageDirectory();
    _filePath = '${directory?.path}/audio_streaming.wav';
    print('Restarting recording, audio saved to: $_filePath');

    await _recorder.startRecorder(
      toFile: _filePath,
      codec: Codec.pcm16WAV,
    );

    await _flutterFft.startRecorder();

    _subscription = _flutterFft.onRecorderStateChanged.listen((data) {
      setState(() {
        _frequency = double.tryParse(data[1].toString()) ?? 0.0;
      });
    });
  }

  Future<void> _stopRecording() async {
    _recordingTimer?.cancel();
    await _recorder.stopRecorder();
    await _flutterFft.stopRecorder();
    _subscription?.cancel();

    setState(() {
      _isRecording = false;
      _isAmplitudeHigh = false;
      _isAmplitudeLow = false;
    });
  }

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
        title: Text('Real-time Sound Analysis'),
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
            Text(
              'Frekuensi: ${_frequency.toStringAsFixed(2)} Hz',
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
            if (_comparisonResult != null)
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Text(
                  _comparisonResult!,
                  style: TextStyle(fontSize: 18, color: Colors.green),
                ),
              ),
          ],
        ),
      ),
    );
  }
}