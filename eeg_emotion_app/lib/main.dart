import 'package:flutter/material.dart';

import 'package:http/http.dart' as http;

import 'dart:convert';

import 'package:file_picker/file_picker.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'EEG Emotion Classifier',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const MyHomePage(title: 'EEG Emotion Classifier'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
// State variables for prediction

  bool _isPredicting = false;

  String _predictionResult = '';

// State variables for patient upload & retraining

  String? _newDatasetVersion;

  bool _isUploading = false;

  bool _isRetraining = false;

// --- PREDICTION (existing feature) ---

  Future<void> _predictEmotion() async {
// Pick EDF file
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      allowedExtensions: ['edf'],
    );
    if (result == null) return;
    setState(() => _isPredicting = rue);
    try {
      var request = http.MultipartRequest(
        'POST',
        Uri.parse(
            'http://localhost:10001/upload'), // adjust if your backend runs on different port
      );
      request.files.add(
        await http.MultipartFile.fromPath('file', result.files.single.path!),
      );
      var response = await request.send();
      var responseBody = await response.stream.bytesToString();
      var json = jsonDecode(responseBody);
      if (response.statusCode == 200) {
        setState(() {
          _predictionResult =
              'Emotion: ${json['emotion']} (${json['confidence']}%)';
        });
        // ignore: use_build_context_synchronously
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Prediction: ${json['emotion']}')),
        );
      } else {
        throw Exception(json['error'] ?? 'Prediction failed');
      }
    } catch (e) {
      // ignore: use_build_context_synchronously
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
      );
    } finally {
      setState(() => _isPredicting = false);
    }
  }

// --- ADD PATIENT DATA (labelled EDF) ---

  Future<void> _addPatient() async {
// Pick EDF file
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      allowedExtensions: ['edf'],
    );
    if (result == null) return;
// Show label selection dialog
    final label = await showDialog<String>(
      // ignore: use_build_context_synchronously
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('Observed Emotion'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: ['POSITIVE', 'NEUTRAL', 'NEGATIVE']
              .map((l) => ListTile(
                    title: Text(l),
                    onTap: () => Navigator.pop(ctx, l),
                  ))
              .toList(),
        ),
      ),
    );
    if (label == null) return;
    setState(() => _isUploading = true);
    var request = http.MultipartRequest(
      'POST',
      Uri.parse('http://localhost:10001/add_patient'),
    );
    request.files.add(
      await http.MultipartFile.fromPath('file', result.files.single.path!),
    );
    request.fields['label'] = label;
    try {
      var response = await request.send();
      var responseBody = await response.stream.bytesToString();
      var json = jsonDecode(responseBody);
      if (response.statusCode == 200) {
        setState(() {
          _newDatasetVersion = json['new_version'];
        });
        // ignore: use_build_context_synchronously
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
              content:
                  Text('Patient added. New version: ${json['new_version']}')),
        );
      } else {
        throw Exception(json['error'] ?? 'Unknown error');
      }
    } catch (e) {
      // ignore: use_build_context_synchronously
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
      );
    } finally {
      setState(() => _isUploading = false);
    }
  }

// --- RETRAIN MODEL on the new dataset version ---

  Future<void> _retrainModel() async {
    if (_newDatasetVersion == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No new patient data added yet')),
      );
      return;
    }
    setState(() => _isRetraining = true);
    try {
      var response = await http.post(
        Uri.parse('http://localhost:10001/retrain'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'version': _newDatasetVersion}),
      );
      var json = jsonDecode(response.body);
      if (response.statusCode == 200) {
        // Retraining succeeded, now reload the model
        var reloadResponse = await http.post(
          Uri.parse('http://localhost:10001/reload_model'),
        );
        var reloadJson = jsonDecode(reloadResponse.body);
        if (reloadResponse.statusCode == 200) {
          // ignore: use_build_context_synchronously
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
                content: Text(
                    'Model retrained and reloaded: ${reloadJson['version']}')),
          );
          setState(() {
            _newDatasetVersion = null; // reset
          });
        } else {
          throw Exception(reloadJson['error'] ?? 'Model reload failed');
        }
      } else {
        throw Exception(json['error'] ?? 'Retraining failed');
      }
    } catch (e) {
      // ignore: use_build_context_synchronously
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
      );
    } finally {
      setState(() => _isRetraining = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              // Prediction button
              ElevatedButton.icon(
                onPressed: _isPredicting ? null : _predictEmotion,
                icon: const Icon(Icons.psychology),
                label: const Text('Predict Emotion'),
              ),
              const SizedBox(height: 20),
              if (_isPredicting) const CircularProgressIndicator(),
              if (_predictionResult.isNotEmpty)
                Padding(
                  padding: const EdgeInsets.only(top: 20),
                  child: Text(_predictionResult,
                      style: const TextStyle(fontSize: 18)),
                ),
              const Divider(height: 40),
              // Add patient data
              ElevatedButton.icon(
                onPressed: _isUploading ? null : _addPatient,
                icon: const Icon(Icons.upload_file),
                label: const Text('Add Patient Data'),
              ),
              const SizedBox(height: 10),
              // Retrain button
              ElevatedButton.icon(
                onPressed: (_isRetraining || _newDatasetVersion == null)
                    ? null
                    : _retrainModel,
                icon: const Icon(Icons.model_training),
                label: const Text('Retrain Model'),
              ),
              if (_isUploading || _isRetraining)
                const Padding(
                  padding: EdgeInsets.only(top: 10),
                  child: CircularProgressIndicator(),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
