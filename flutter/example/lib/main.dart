import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:odometer_sdk/odometer_sdk.dart';

void main() => runApp(const OdometerDemoApp());

class OdometerDemoApp extends StatelessWidget {
  const OdometerDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Odometer SDK Demo',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: const OdometerReaderScreen(),
    );
  }
}

class OdometerReaderScreen extends StatefulWidget {
  const OdometerReaderScreen({super.key});

  @override
  State<OdometerReaderScreen> createState() => _OdometerReaderScreenState();
}

class _OdometerReaderScreenState extends State<OdometerReaderScreen> {
  final ImagePicker _picker = ImagePicker();

  String? _imagePath;
  OdometerResult? _result;
  bool _loading = false;
  String? _error;

  // In a real app, load this from local storage / your backend after a
  // successful read, keyed by vehicle ID.
  int? _previousReading = 45210;

  Future<void> _captureAndRead() async {
    final photo = await _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 95, // keep quality high - downscaling hurts recognition
    );
    if (photo == null) return;

    setState(() {
      _imagePath = photo.path;
      _loading = true;
      _error = null;
      _result = null;
    });

    try {
      final result = await OdometerSdk.read(
        photo.path,
        previousReading: _previousReading,
      );
      setState(() => _result = result);

      // Handle the confidence-driven action explicitly - this is the core
      // usage pattern the SDK is designed around.
      switch (result.action) {
        case ConfidenceAction.accept:
          if (result.isValid) {
            // Good reading, passed business rules too - safe to save.
            setState(() => _previousReading = int.parse(result.reading!));
          }
          break;
        case ConfidenceAction.retryPreprocessing:
          // The SDK itself already tried its default enhancement settings;
          // this signals the app could re-run with alternate settings or
          // simply nudge the user ("having trouble reading this, try again
          // with better lighting") rather than silently accepting a
          // borderline result.
          break;
        case ConfidenceAction.askRetake:
          // Confidence too low to trust - always re-prompt rather than
          // showing an uncertain number as if it were reliable.
          break;
      }
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Odometer Reader')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            if (_previousReading != null)
              Text('Previous reading: $_previousReading km',
                  style: Theme.of(context).textTheme.bodyMedium),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _loading ? null : _captureAndRead,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Take odometer photo'),
            ),
            const SizedBox(height: 24),
            if (_loading) const Center(child: CircularProgressIndicator()),
            if (_error != null)
              Text('Error: $_error', style: const TextStyle(color: Colors.red)),
            if (_result != null) _ResultCard(result: _result!),
          ],
        ),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final OdometerResult result;
  const _ResultCard({required this.result});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            switch (result.action) {
              ConfidenceAction.accept when result.isValid => Text(
                  '${result.reading} km',
                  style: Theme.of(context)
                      .textTheme
                      .headlineMedium
                      ?.copyWith(color: Colors.green.shade700, fontWeight: FontWeight.bold),
                ),
              ConfidenceAction.accept => Text(
                  'Read "${result.reading}" but flagged: ${result.validationMessage}',
                  style: TextStyle(color: Colors.orange.shade800),
                ),
              ConfidenceAction.retryPreprocessing => Text(
                  'Uncertain read ("${result.reading}") — retrying with better processing recommended',
                  style: TextStyle(color: Colors.orange.shade800),
                ),
              ConfidenceAction.askRetake => const Text(
                  'Could not get a confident reading — please retake the photo',
                  style: TextStyle(color: Colors.red),
                ),
            },
            const SizedBox(height: 8),
            Text('Confidence: ${result.confidence.toStringAsFixed(1)}%'),
            if (result.issues.isNotEmpty)
              Text('Issues: ${result.issues.join(", ")}',
                  style: const TextStyle(fontSize: 12, color: Colors.grey)),
            if (result.usedMockModels)
              const Padding(
                padding: EdgeInsets.only(top: 8),
                child: Text(
                  '⚠️ Running in mock mode — no trained ONNX models bundled yet',
                  style: TextStyle(fontSize: 12, fontStyle: FontStyle.italic),
                ),
              ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 4,
              children: result.digitConfidences
                  .map((c) => Chip(
                        label: Text('${(c * 100).toStringAsFixed(0)}%'),
                        backgroundColor: c > 0.97
                            ? Colors.green.shade50
                            : c > 0.80
                                ? Colors.orange.shade50
                                : Colors.red.shade50,
                      ))
                  .toList(),
            ),
          ],
        ),
      ),
    );
  }
}
