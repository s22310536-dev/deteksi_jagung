import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:web/web.dart' as web;

void main() => runApp(
  const MaterialApp(debugShowCheckedModeBanner: false, home: CornVisionPage()),
);

enum _InteractionMode { none, crop, gt }

class NormBox {
  final double ymin;
  final double xmin;
  final double ymax;
  final double xmax;

  const NormBox({
    required this.ymin,
    required this.xmin,
    required this.ymax,
    required this.xmax,
  });

  double get width => math.max(0.0, xmax - xmin);
  double get height => math.max(0.0, ymax - ymin);
  double get area => width * height;

  double iouWith(NormBox other) {
    final interLeft = math.max(xmin, other.xmin);
    final interTop = math.max(ymin, other.ymin);
    final interRight = math.min(xmax, other.xmax);
    final interBottom = math.min(ymax, other.ymax);

    final interW = math.max(0.0, interRight - interLeft);
    final interH = math.max(0.0, interBottom - interTop);
    final interArea = interW * interH;
    final unionArea = area + other.area - interArea;

    if (unionArea <= 0) return 0.0;
    return interArea / unionArea;
  }
}

class CvDetection {
  final String label;
  final double confidence;
  final String fiturVisual;
  final NormBox box;
  final bool isManualGt;

  const CvDetection({
    required this.label,
    required this.confidence,
    required this.fiturVisual,
    required this.box,
    this.isManualGt = false,
  });

  CvDetection copyWith({String? fiturVisual}) {
    return CvDetection(
      label: label,
      confidence: confidence,
      fiturVisual: fiturVisual ?? this.fiturVisual,
      box: box,
      isManualGt: isManualGt,
    );
  }
}

class CornVisionPage extends StatefulWidget {
  const CornVisionPage({super.key});

  @override
  State<CornVisionPage> createState() => _CornVisionPageState();
}

class _CornVisionPageState extends State<CornVisionPage> {
  static const String _apiKey = '';

  Uint8List? _originalBytes;
  int _origWidth = 0;
  int _origHeight = 0;

  Uint8List? _imageBytes;
  int _imgWidth = 0;
  int _imgHeight = 0;
  bool _isCropped = false;

  String _mimeType = 'image/jpeg';
  String _fileName = '';
  bool _canPreview = true;
  bool _isLoading = false;

  _InteractionMode _mode = _InteractionMode.none;
  final List<NormBox> _gtBoxes = [];
  Offset? _dragStartNorm;
  Offset? _dragCurrentNorm;

  String _message =
      'Unggah foto daun atau tongkol jagung untuk analisa Computer Vision.';
  Map<String, dynamic>? _analysis;
  List<CvDetection> _detections = [];
  bool _isManualRoiReport = false;

  Future<({Uint8List bytes, int width, int height})?> _convertToJpeg(
    String srcUrl,
  ) async {
    final completer = Completer<({Uint8List bytes, int width, int height})?>();
    final img = web.HTMLImageElement();

    img.addEventListener(
      'load',
      ((web.Event _) {
        try {
          if (img.naturalWidth == 0 || img.naturalHeight == 0) {
            if (!completer.isCompleted) completer.complete(null);
            return;
          }
          final canvas = web.HTMLCanvasElement()
            ..width = img.naturalWidth
            ..height = img.naturalHeight;
          final ctx = canvas.getContext('2d') as web.CanvasRenderingContext2D?;
          if (ctx == null) {
            if (!completer.isCompleted) completer.complete(null);
            return;
          }
          ctx.drawImage(img, 0, 0);
          final dataUrl = canvas.toDataURL('image/jpeg');
          final base64Str = dataUrl.split(',').last;
          if (!completer.isCompleted) {
            completer.complete((
              bytes: base64Decode(base64Str),
              width: img.naturalWidth,
              height: img.naturalHeight,
            ));
          }
        } catch (_) {
          if (!completer.isCompleted) completer.complete(null);
        }
      }).toJS,
    );

    img.addEventListener(
      'error',
      ((web.Event _) {
        if (!completer.isCompleted) completer.complete(null);
      }).toJS,
    );

    img.src = srcUrl;
    return completer.future;
  }

  Future<({Uint8List bytes, int width, int height})?> _cropCurrentImage(
    NormBox box,
  ) async {
    if (_imageBytes == null || _imgWidth <= 0 || _imgHeight <= 0) return null;

    final sx = ((box.xmin / 1000.0) * _imgWidth)
        .round()
        .clamp(0, _imgWidth - 1)
        .toInt();
    final sy = ((box.ymin / 1000.0) * _imgHeight)
        .round()
        .clamp(0, _imgHeight - 1)
        .toInt();
    final ex = ((box.xmax / 1000.0) * _imgWidth)
        .round()
        .clamp(sx + 1, _imgWidth)
        .toInt();
    final ey = ((box.ymax / 1000.0) * _imgHeight)
        .round()
        .clamp(sy + 1, _imgHeight)
        .toInt();

    final sw = ex - sx;
    final sh = ey - sy;
    if (sw < 20 || sh < 20) return null;

    final completer = Completer<({Uint8List bytes, int width, int height})?>();
    final img = web.HTMLImageElement();

    img.addEventListener(
      'load',
      ((web.Event _) {
        try {
          final canvas = web.HTMLCanvasElement()
            ..width = sw
            ..height = sh;
          final ctx = canvas.getContext('2d') as web.CanvasRenderingContext2D?;
          if (ctx == null) {
            if (!completer.isCompleted) completer.complete(null);
            return;
          }
          ctx.drawImage(img, -sx, -sy);
          final dataUrl = canvas.toDataURL('image/jpeg');
          final base64Str = dataUrl.split(',').last;
          if (!completer.isCompleted) {
            completer.complete((
              bytes: base64Decode(base64Str),
              width: sw,
              height: sh,
            ));
          }
        } catch (_) {
          if (!completer.isCompleted) completer.complete(null);
        }
      }).toJS,
    );

    img.addEventListener(
      'error',
      ((web.Event _) {
        if (!completer.isCompleted) completer.complete(null);
      }).toJS,
    );

    img.src = 'data:image/jpeg;base64,${base64Encode(_imageBytes!)}';
    return completer.future;
  }

  String _guessMimeType(String name, String? fallback) {
    if (fallback != null && fallback.isNotEmpty) return fallback;
    final lower = name.toLowerCase();
    if (lower.endsWith('.png')) return 'image/png';
    if (lower.endsWith('.webp')) return 'image/webp';
    if (lower.endsWith('.heic')) return 'image/heic';
    if (lower.endsWith('.heif')) return 'image/heif';
    return 'image/jpeg';
  }

  Future<void> _pickImage() async {
    final picker = ImagePicker();
    final XFile? file = await picker.pickImage(source: ImageSource.gallery);
    if (file == null) return;

    final converted = await _convertToJpeg(file.path);

    if (converted != null) {
      setState(() {
        _originalBytes = converted.bytes;
        _origWidth = converted.width;
        _origHeight = converted.height;
        _imageBytes = converted.bytes;
        _imgWidth = converted.width;
        _imgHeight = converted.height;
        _isCropped = false;
        _mimeType = 'image/jpeg';
        _fileName = file.name;
        _canPreview = true;
        _analysis = null;
        _detections = [];
        _isManualRoiReport = false;
        _gtBoxes.clear();
        _mode = _InteractionMode.none;
        _message =
            'Citra siap (${file.name} — ${_imgWidth}x$_imgHeight px). Klik "Tandai GT (Hijau)" untuk menandai lubang secara manual lalu klik "Analisa Kotak Hijau", atau langsung klik "Analisa CV".';
      });
    } else {
      final rawBytes = await file.readAsBytes();
      setState(() {
        _originalBytes = rawBytes;
        _origWidth = 0;
        _origHeight = 0;
        _imageBytes = rawBytes;
        _imgWidth = 0;
        _imgHeight = 0;
        _isCropped = false;
        _mimeType = _guessMimeType(file.name, file.mimeType);
        _fileName = file.name;
        _canPreview = false;
        _analysis = null;
        _detections = [];
        _isManualRoiReport = false;
        _gtBoxes.clear();
        _mode = _InteractionMode.none;
        _message = 'Citra dimuat (${file.name}). Tekan Analisa CV.';
      });
    }
  }

  void _resetCrop() {
    if (_originalBytes == null) return;
    setState(() {
      _imageBytes = _originalBytes;
      _imgWidth = _origWidth;
      _imgHeight = _origHeight;
      _isCropped = false;
      _analysis = null;
      _detections = [];
      _isManualRoiReport = false;
      _gtBoxes.clear();
      _mode = _InteractionMode.none;
      _message =
          'Foto dikembalikan ke ukuran asli (${_imgWidth}x$_imgHeight px).';
    });
  }

  Uri _buildGeminiUrl() {
    return _apiKey.startsWith('AQ.')
        ? Uri.parse(
            'https://aiplatform.googleapis.com/v1/publishers/google/models/gemini-2.5-flash:generateContent?key=$_apiKey',
          )
        : Uri.parse(
            'https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$_apiKey',
          );
  }

  Future<String> _callGeminiWithRetry(
    String prompt, {
    double temperature = 0.1,
    required String stageLabel,
  }) async {
    final url = _buildGeminiUrl();
    final payload = jsonEncode({
      'contents': [
        {
          'role': 'user',
          'parts': [
            {'text': prompt},
            {
              'inlineData': {
                'mimeType': _mimeType,
                'data': base64Encode(_imageBytes!),
              },
            },
          ],
        },
      ],
      'generationConfig': {
        'responseMimeType': 'application/json',
        'temperature': temperature,
      },
    });

    const maxAttempts = 3;
    for (var attempt = 1; attempt <= maxAttempts; attempt++) {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: payload,
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final rawText =
            data['candidates']?[0]?['content']?['parts']?[0]?['text']
                as String?;
        if (rawText == null || rawText.trim().isEmpty) {
          throw Exception('Respon AI kosong pada $stageLabel.');
        }
        return rawText
            .replaceAll(RegExp(r'^```json\s*', multiLine: true), '')
            .replaceAll(RegExp(r'^```\s*', multiLine: true), '')
            .trim();
      }

      if ((response.statusCode == 503 || response.statusCode == 429) &&
          attempt < maxAttempts) {
        if (mounted) {
          setState(() {
            _message =
                'Server AI sedang padat (${response.statusCode}). Mencoba ulang $stageLabel ($attempt/${maxAttempts - 1})...';
          });
        }
        await Future.delayed(const Duration(seconds: 3));
        continue;
      }

      throw Exception(
        'Gagal $stageLabel (${response.statusCode}): ${response.body}',
      );
    }

    throw Exception('Gagal menghubungi server AI setelah beberapa percobaan.');
  }

  List<CvDetection> _filterAndDeduplicate(List<CvDetection> raw) {
    final maxArea = _isCropped ? 550000.0 : 280000.0;
    final filtered = raw.where((d) {
      return d.box.width >= 8 && d.box.height >= 8 && d.box.area <= maxArea;
    }).toList();

    filtered.sort((a, b) => b.confidence.compareTo(a.confidence));
    final List<CvDetection> kept = [];
    for (final candidate in filtered) {
      final overlapsExisting = kept.any(
        (existing) => existing.box.iouWith(candidate.box) > 0.65,
      );
      if (!overlapsExisting) {
        kept.add(candidate);
      }
    }
    return kept;
  }

  /// Mode 1: Analisa otomatis penuh (AI mencari kotak oranye sendiri)
  Future<void> _analyzeCorn() async {
    if (_imageBytes == null) return;

    if (_apiKey.isEmpty) {
      setState(() {
        _message = 'API Key belum diisi di baris _apiKey dalam lib/main.dart.';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _mode = _InteractionMode.none;
      _analysis = null;
      _detections = [];
      _isManualRoiReport = false;
      _message = 'Tahap 1/2: Mendeteksi koordinat lubang & gigitan hama (Object Detection)...';
    });

    const detectionPrompt = '''
Detect every visible pest hole, chewed leaf margin/bite mark, transparent epidermis feeding scar, or insect/larva ONLY on the main foreground corn leaf or corn cob in this image.

Strict rules:
- Focus ONLY on the closest, sharpest foreground corn leaf or cob.
- Completely IGNORE background plants, soil, dry brown stalks behind the leaf, shadows, and empty gaps between different leaves.
- Draw TIGHT bounding boxes strictly around each individual hole or bite mark. NEVER draw one big box around the whole leaf or background.
- If the image is not corn or has no pest holes/damage, return [].

Return ONLY a JSON array with this exact format:
[
  {
    "box_2d": [ymin, xmin, ymax, xmax],
    "label": "Lubang Hama",
    "confidence": 0.85
  }
]

Allowed labels: "Lubang Hama", "Tepi Tergigit", "Gerekan Epidermis", "Hama Ulat", "Bercak Penyakit".
"box_2d" must use integer coordinates from 0 to 1000 in [ymin, xmin, ymax, xmax] order.
''';

    try {
      final stage1Json = await _callGeminiWithRetry(
        detectionPrompt,
        temperature: 0.0,
        stageLabel: 'Tahap 1 (Deteksi Kotak)',
      );

      final decodedStage1 = jsonDecode(stage1Json);
      final List rawItems = decodedStage1 is List
          ? decodedStage1
          : (decodedStage1 is Map && decodedStage1['deteksi'] is List
                ? decodedStage1['deteksi'] as List
                : []);

      final List<CvDetection> parsedBoxes = [];
      for (final item in rawItems) {
        if (item is Map && item['box_2d'] is List) {
          final coords = item['box_2d'] as List;
          if (coords.length == 4) {
            final ymin = (coords[0] as num)
                .toDouble()
                .clamp(0.0, 1000.0)
                .toDouble();
            final xmin = (coords[1] as num)
                .toDouble()
                .clamp(0.0, 1000.0)
                .toDouble();
            final ymax = (coords[2] as num)
                .toDouble()
                .clamp(0.0, 1000.0)
                .toDouble();
            final xmax = (coords[3] as num)
                .toDouble()
                .clamp(0.0, 1000.0)
                .toDouble();

            parsedBoxes.add(
              CvDetection(
                label: (item['label'] ?? 'Lubang Hama').toString(),
                confidence: ((item['confidence'] as num?)?.toDouble() ?? 0.80)
                    .clamp(0.0, 1.0)
                    .toDouble(),
                fiturVisual: '-',
                box: NormBox(
                  ymin: math.min(ymin, ymax),
                  xmin: math.min(xmin, xmax),
                  ymax: math.max(ymin, ymax),
                  xmax: math.max(xmin, xmax),
                ),
              ),
            );
          }
        }
      }

      final cleanBoxes = _filterAndDeduplicate(parsedBoxes);

      if (mounted) {
        setState(() {
          _detections = cleanBoxes;
          _message =
              'Tahap 2/2: Ditemukan ${cleanBoxes.length} ROI. Menyusun analisis fitur Computer Vision...';
        });
      }

      final roiSummaryForPrompt = List.generate(cleanBoxes.length, (i) {
        final b = cleanBoxes[i].box;
        return {
          'roi_index': i + 1,
          'label': cleanBoxes[i].label,
          'box_2d': [
            b.ymin.round(),
            b.xmin.round(),
            b.ymax.round(),
            b.xmax.round(),
          ],
        };
      });

      final reportPrompt =
          '''
Bertindaklah sebagai sistem Computer Vision (Visual Defect Inspection) untuk tanaman jagung.
Pada tahap lokalisasi sebelumnya, sistem menandai ${cleanBoxes.length} ROI berikut (koordinat 0-1000 [ymin, xmin, ymax, xmax]):
${jsonEncode(roiSummaryForPrompt)}

Buat laporan teknis Computer Vision dalam Bahasa Indonesia berdasarkan citra jagung utama (foreground) dan daftar ROI di atas.
Kembalikan HANYA JSON valid dengan struktur persis berikut:
{
  "bagian_objek": "Daun Jagung / Tongkol Jagung / Biji Jagung / Bukan Jagung",
  "status": "SEHAT (BEBAS HAMA)" | "TERDETEKSI HAMA RINGAN" | "TERDETEKSI HAMA BERAT" | "BERCAK PENYAKIT (NON-HAMA)" | "BUKAN JAGUNG",
  "ringkasan_cv": "penjelasan teknis singkat hasil inspeksi visual dan lokalisasi kerusakan pada objek utama",
  "ekstraksi_fitur": {
    "fitur_warna": "analisis kontras warna area sehat vs area lubang/gigitan pada daun atau tongkol utama",
    "fitur_morfologi": "analisis bentuk kontur lubang, sobekan tepi, atau pola gerekan",
    "catatan_false_positive": "penjelasan cara sistem membedakan lubang hama dari celah daun, latar belakang, bayangan, atau bercak penyakit"
  },
  "dugaan_hama": "misal: Ulat Grayak (Spodoptera frugiperda), Ulat Tongkol, Kumbang Bubuk, atau Tidak terdeteksi hama",
  "estimasi_area_rusak_persen": "misal: ~8% dari permukaan daun utama",
  "rekomendasi": "saran tindak lanjut pengendalian hama atau sortir",
  "ciri_per_roi": [
    "deskripsi singkat ciri visual untuk ROI #1 sesuai urutan daftar ROI di atas"
  ]
}
''';

      final stage2Json = await _callGeminiWithRetry(
        reportPrompt,
        temperature: 0.2,
        stageLabel: 'Tahap 2 (Laporan CV)',
      );

      final parsedReport = jsonDecode(stage2Json) as Map<String, dynamic>;
      final roiDescs = (parsedReport['ciri_per_roi'] as List?) ?? [];

      final enrichedBoxes = List.generate(cleanBoxes.length, (i) {
        final desc = i < roiDescs.length
            ? roiDescs[i].toString()
            : 'Kerusakan/lubang terdeteksi pada area daun utama.';
        return cleanBoxes[i].copyWith(fiturVisual: desc);
      });

      if (mounted) {
        setState(() {
          _analysis = parsedReport;
          _detections = enrichedBoxes;
          _isManualRoiReport = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _message = e.toString().replaceFirst('Exception: ', ''));
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  /// Mode 2 (Human-in-the-Loop): Manusia menandai kotak hijau, AI menganalisis isinya
  Future<void> _analyzeGtBoxes() async {
    if (_imageBytes == null || _gtBoxes.isEmpty) return;

    if (_apiKey.isEmpty) {
      setState(() {
        _message = 'API Key belum diisi di baris _apiKey dalam lib/main.dart.';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _mode = _InteractionMode.none;
      _analysis = null;
      _detections = [];
      _isManualRoiReport = true;
      _message =
          'Menganalisis ${_gtBoxes.length} area Kotak Hijau yang kamu tandai...';
    });

    final gtCoordsForPrompt = List.generate(_gtBoxes.length, (i) {
      final b = _gtBoxes[i];
      return {
        'gt_index': i + 1,
        'box_2d': [
          b.ymin.round(),
          b.xmin.round(),
          b.ymax.round(),
          b.xmax.round(),
        ],
      };
    });

    final manualRoiPrompt =
        '''
Bertindaklah sebagai sistem Computer Vision (Human-in-the-Loop ROI Diagnosis) untuk tanaman jagung.
Pengguna telah menandai ${_gtBoxes.length} area spesifik (Kotak Hijau / Manual ROI) pada foto ini dengan koordinat skala 0-1000 [ymin, xmin, ymax, xmax]:
${jsonEncode(gtCoordsForPrompt)}

Tugasmu: FOKUS 100% menganalisis kondisi di dalam koordinat kotak-kotak hijau tersebut (jangan mencari objek lain di latar belakang). Tentukan apakah isi tiap kotak tersebut adalah lubang gigitan hama, tepi tergigit, gerekan epidermis, ulat/serangga, bercak penyakit, atau daun sehat.

Kembalikan HANYA JSON valid dengan struktur persis berikut:
{
  "bagian_objek": "Daun Jagung / Tongkol Jagung / Biji Jagung / Bukan Jagung",
  "status": "SEHAT (BEBAS HAMA)" | "TERDETEKSI HAMA RINGAN" | "TERDETEKSI HAMA BERAT" | "BERCAK PENYAKIT (NON-HAMA)" | "BUKAN JAGUNG",
  "ringkasan_cv": "penjelasan teknis hasil diagnosa AI terhadap area kotak hijau yang ditandai pengguna",
  "ekstraksi_fitur": {
    "fitur_warna": "analisis kontras warna pada area di dalam kotak hijau",
    "fitur_morfologi": "analisis bentuk lubang, sobekan tepi, atau pola gerekan di dalam kotak hijau",
    "catatan_false_positive": "verifikasi apakah area di dalam kotak hijau benar-benar kerusakan hama atau hanya bayangan/bercak penyakit"
  },
  "dugaan_hama": "misal: Ulat Grayak (Spodoptera frugiperda), Ulat Tongkol, Kumbang Bubuk, atau Tidak terdeteksi hama",
  "estimasi_area_rusak_persen": "misal: ~10% dari permukaan daun utama",
  "rekomendasi": "saran tindak lanjut pengendalian hama atau sortir",
  "hasil_per_gt": [
    {
      "label": "Lubang Hama / Tepi Tergigit / Gerekan Epidermis / Hama Ulat / Bercak Penyakit / Sehat",
      "confidence": 0.90,
      "fitur_visual": "penjelasan detail ciri visual kerusakan tepat di dalam kotak hijau ini"
    }
  ]
}

Pastikan jumlah elemen di "hasil_per_gt" SAMA PERSIS dengan jumlah kotak hijau (${_gtBoxes.length}) dan berurutan dari GT #1 sampai GT #${_gtBoxes.length}.
''';

    try {
      final resultJson = await _callGeminiWithRetry(
        manualRoiPrompt,
        temperature: 0.15,
        stageLabel: 'Analisa Kotak Hijau',
      );

      final parsed = jsonDecode(resultJson) as Map<String, dynamic>;
      final rawPerGt = (parsed['hasil_per_gt'] as List?) ?? [];

      final List<CvDetection> gtDiagnoses = List.generate(_gtBoxes.length, (i) {
        final item = i < rawPerGt.length && rawPerGt[i] is Map
            ? rawPerGt[i] as Map
            : const {};
        return CvDetection(
          label: (item['label'] ?? 'Area Terdiagnosa').toString(),
          confidence: ((item['confidence'] as num?)?.toDouble() ?? 0.85)
              .clamp(0.0, 1.0)
              .toDouble(),
          fiturVisual:
              (item['fitur_visual'] ??
                      'Area kotak hijau telah dianalisis oleh sistem.')
                  .toString(),
          box: _gtBoxes[i],
          isManualGt: true,
        );
      });

      if (mounted) {
        setState(() {
          _analysis = parsed;
          _detections = gtDiagnoses;
          _isManualRoiReport = true;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _message = e.toString().replaceFirst('Exception: ', ''));
      }
    } finally {
      if (mounted) {
        setState(() => _isLoading = false);
      }
    }
  }

  Color _statusColor(String status) {
    final s = status.toUpperCase();
    if (s.contains('SEHAT')) return Colors.green.shade700;
    if (s.contains('RINGAN') || s.contains('BERCAK')) {
      return Colors.orange.shade800;
    }
    if (s.contains('BERAT')) return Colors.red.shade700;
    return Colors.grey.shade700;
  }

  String _formatPixelCoords(NormBox b) {
    if (_imgWidth <= 0 || _imgHeight <= 0) {
      return 'Norm [ymin:${b.ymin.round()}, xmin:${b.xmin.round()}, ymax:${b.ymax.round()}, xmax:${b.xmax.round()}]';
    }
    final x = ((b.xmin / 1000) * _imgWidth).round();
    final y = ((b.ymin / 1000) * _imgHeight).round();
    final w = ((b.width / 1000) * _imgWidth).round();
    final h = ((b.height / 1000) * _imgHeight).round();
    return 'x=$x, y=$y, w=${w}px, h=${h}px';
  }

  Widget _buildGtEvaluationCard() {
    // Tampilkan evaluasi IoU hanya saat membandingkan GT Hijau vs Prediksi Oranye
    if (_gtBoxes.isEmpty || _isManualRoiReport) {
      return const SizedBox.shrink();
    }
    final autoPreds = _detections.where((d) => !d.isManualGt).toList();
    if (autoPreds.isEmpty) return const SizedBox.shrink();

    int tp = 0;
    int lowIou = 0;
    int fn = 0;
    final matchedPredIndices = <int>{};
    final List<String> details = [];

    for (var i = 0; i < _gtBoxes.length; i++) {
      final gt = _gtBoxes[i];
      double bestIou = 0.0;
      int bestPredIdx = -1;

      for (var j = 0; j < autoPreds.length; j++) {
        final iou = gt.iouWith(autoPreds[j].box);
        if (iou > bestIou) {
          bestIou = iou;
          bestPredIdx = j;
        }
      }

      if (bestIou >= 0.5 && bestPredIdx != -1) {
        tp++;
        matchedPredIndices.add(bestPredIdx);
        details.add(
          'GT #${i + 1} vs Prediksi #${bestPredIdx + 1}: IoU = ${(bestIou * 100).toStringAsFixed(1)}% (True Positive — Akurat)',
        );
      } else if (bestIou > 0 && bestPredIdx != -1) {
        lowIou++;
        matchedPredIndices.add(bestPredIdx);
        details.add(
          'GT #${i + 1} vs Prediksi #${bestPredIdx + 1}: IoU = ${(bestIou * 100).toStringAsFixed(1)}% (Lokalisasi Meleset / IoU < 50%)',
        );
      } else {
        fn++;
        details.add(
          'GT #${i + 1}: IoU = 0.0% (False Negative — Tidak terdeteksi model)',
        );
      }
    }

    final fp = autoPreds.length - matchedPredIndices.length;

    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.teal.shade50,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.teal.shade600, width: 1.2),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.analytics, color: Colors.teal.shade800, size: 20),
              const SizedBox(width: 8),
              Text(
                'Evaluasi GT (Hijau) vs Prediksi (Oranye)',
                style: TextStyle(
                  fontWeight: FontWeight.bold,
                  color: Colors.teal.shade900,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 6,
            children: [
              _metricChip('TP (IoU ≥ 0.5): $tp', Colors.green.shade700),
              _metricChip('Low IoU: $lowIou', Colors.orange.shade800),
              _metricChip('FN (Terlewat): $fn', Colors.red.shade700),
              _metricChip('FP (Salah Deteksi): $fp', Colors.deepPurple),
            ],
          ),
          const SizedBox(height: 10),
          ...details.map(
            (line) => Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: SelectableText(
                '• $line',
                style: const TextStyle(fontSize: 13),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _metricChip(String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: color,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Text(
        text,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 12,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }

  Widget _buildSectionItem(String label, String content) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 10),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: const TextStyle(
              fontWeight: FontWeight.bold,
              fontSize: 13,
              color: Colors.black54,
            ),
          ),
          const SizedBox(height: 2),
          SelectableText(content, style: const TextStyle(fontSize: 14)),
        ],
      ),
    );
  }

  Widget _buildImageViewer() {
    if (_imageBytes == null) {
      return Container(
        height: 260,
        alignment: Alignment.center,
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade400),
          borderRadius: BorderRadius.circular(12),
        ),
        child: const Text('Belum ada citra dipilih'),
      );
    }

    if (!_canPreview || _imgWidth <= 0 || _imgHeight <= 0) {
      return Container(
        height: 240,
        alignment: Alignment.center,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade400),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.image_not_supported, size: 42),
            const SizedBox(height: 8),
            Text(
              'Pratinjau $_fileName tidak didukung browser,\ntapi citra tetap siap dianalisa.',
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    final aspectRatio = _imgWidth / _imgHeight;
    final borderColor = _mode == _InteractionMode.crop
        ? Colors.blue.shade700
        : _mode == _InteractionMode.gt
        ? Colors.teal.shade600
        : Colors.grey.shade400;

    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: Colors.grey.shade50,
        border: Border.all(
          color: borderColor,
          width: _mode != _InteractionMode.none ? 2.2 : 1,
        ),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        children: [
          Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxHeight: 400),
              child: AspectRatio(
                aspectRatio: aspectRatio,
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final w = constraints.maxWidth;
                    final h = constraints.maxHeight;

                    NormBox? activeDragBox;
                    if (_dragStartNorm != null && _dragCurrentNorm != null) {
                      activeDragBox = NormBox(
                        xmin: math.min(
                          _dragStartNorm!.dx,
                          _dragCurrentNorm!.dx,
                        ),
                        ymin: math.min(
                          _dragStartNorm!.dy,
                          _dragCurrentNorm!.dy,
                        ),
                        xmax: math.max(
                          _dragStartNorm!.dx,
                          _dragCurrentNorm!.dx,
                        ),
                        ymax: math.max(
                          _dragStartNorm!.dy,
                          _dragCurrentNorm!.dy,
                        ),
                      );
                    }

                    final isInteractive = _mode != _InteractionMode.none;

                    return GestureDetector(
                      onPanStart: isInteractive
                          ? (details) {
                              final nx = (details.localPosition.dx / w * 1000)
                                  .clamp(0.0, 1000.0)
                                  .toDouble();
                              final ny = (details.localPosition.dy / h * 1000)
                                  .clamp(0.0, 1000.0)
                                  .toDouble();
                              setState(() {
                                _dragStartNorm = Offset(nx, ny);
                                _dragCurrentNorm = Offset(nx, ny);
                              });
                            }
                          : null,
                      onPanUpdate: isInteractive
                          ? (details) {
                              final nx = (details.localPosition.dx / w * 1000)
                                  .clamp(0.0, 1000.0)
                                  .toDouble();
                              final ny = (details.localPosition.dy / h * 1000)
                                  .clamp(0.0, 1000.0)
                                  .toDouble();
                              setState(() => _dragCurrentNorm = Offset(nx, ny));
                            }
                          : null,
                      onPanEnd: isInteractive
                          ? (_) async {
                              if (_dragStartNorm != null &&
                                  _dragCurrentNorm != null) {
                                final box = NormBox(
                                  xmin: math.min(
                                    _dragStartNorm!.dx,
                                    _dragCurrentNorm!.dx,
                                  ),
                                  ymin: math.min(
                                    _dragStartNorm!.dy,
                                    _dragCurrentNorm!.dy,
                                  ),
                                  xmax: math.max(
                                    _dragStartNorm!.dx,
                                    _dragCurrentNorm!.dx,
                                  ),
                                  ymax: math.max(
                                    _dragStartNorm!.dy,
                                    _dragCurrentNorm!.dy,
                                  ),
                                );

                                if (box.width > 12 && box.height > 12) {
                                  if (_mode == _InteractionMode.gt) {
                                    setState(() => _gtBoxes.add(box));
                                  } else if (_mode == _InteractionMode.crop) {
                                    final cropped = await _cropCurrentImage(
                                      box,
                                    );
                                    if (cropped != null && mounted) {
                                      setState(() {
                                        _imageBytes = cropped.bytes;
                                        _imgWidth = cropped.width;
                                        _imgHeight = cropped.height;
                                        _isCropped = true;
                                        _mode = _InteractionMode.none;
                                        _analysis = null;
                                        _detections = [];
                                        _isManualRoiReport = false;
                                        _gtBoxes.clear();
                                        _message =
                                            'Area daun dipotong (${_imgWidth}x$_imgHeight px). Klik Analisa CV atau Tandai GT (Hijau).';
                                      });
                                    }
                                  }
                                }
                              }
                              if (mounted) {
                                setState(() {
                                  _dragStartNorm = null;
                                  _dragCurrentNorm = null;
                                });
                              }
                            }
                          : null,
                      child: MouseRegion(
                        cursor: isInteractive
                            ? SystemMouseCursors.precise
                            : SystemMouseCursors.basic,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            Image.memory(_imageBytes!, fit: BoxFit.fill),
                            CustomPaint(
                              painter: _CvOverlayPainter(
                                detections: _detections,
                                gtBoxes: _gtBoxes,
                                activeDraft: activeDragBox,
                                isCropDraft: _mode == _InteractionMode.crop,
                              ),
                            ),
                          ],
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),
          ),
          const SizedBox(height: 10),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              _legendItem(Colors.teal.shade600, 'Hijau: Kotak Manual / GT'),
              const SizedBox(width: 16),
              _legendItem(Colors.deepOrange, 'Oranye: Prediksi Otomatis AI'),
            ],
          ),
          if (_mode == _InteractionMode.crop)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(
                'Klik & geser mouse untuk memotong area daun utama.',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.blue.shade800,
                  fontWeight: FontWeight.w600,
                ),
              ),
            )
          else if (_mode == _InteractionMode.gt)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(
                'Klik & geser mouse tepat di lubang hama, lalu klik "Analisa Kotak Hijau".',
                style: TextStyle(
                  fontSize: 12,
                  color: Colors.teal.shade800,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _legendItem(Color color, String label) {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        Container(
          width: 14,
          height: 14,
          decoration: BoxDecoration(
            border: Border.all(color: color, width: 2.5),
            borderRadius: BorderRadius.circular(2),
          ),
        ),
        const SizedBox(width: 6),
        Text(label, style: const TextStyle(fontSize: 12)),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final status = _analysis?['status']?.toString() ?? '';
    final fitur = (_analysis?['ekstraksi_fitur'] as Map?) ?? {};
    final roiAccent = _isManualRoiReport
        ? Colors.teal.shade700
        : Colors.deepOrange;
    final roiBg = _isManualRoiReport
        ? Colors.teal.shade50
        : Colors.orange.shade50;
    final roiBorder = _isManualRoiReport
        ? Colors.teal.shade400
        : Colors.orange.shade300;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Computer Vision — Deteksi Hama Jagung'),
      ),
      body: Center(
        child: SingleChildScrollView(
          physics: _mode != _InteractionMode.none
              ? const NeverScrollableScrollPhysics()
              : null,
          padding: const EdgeInsets.all(24),
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 680),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                _buildImageViewer(),
                const SizedBox(height: 14),
                Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: [
                    OutlinedButton.icon(
                      onPressed: _isLoading ? null : _pickImage,
                      icon: const Icon(Icons.upload_file),
                      label: const Text('Pilih Gambar'),
                    ),
                    if (_imageBytes != null && _canPreview)
                      FilterChip(
                        selected: _mode == _InteractionMode.crop,
                        selectedColor: Colors.blue.shade100,
                        checkmarkColor: Colors.blue.shade900,
                        avatar: Icon(
                          Icons.crop,
                          size: 18,
                          color: Colors.blue.shade800,
                        ),
                        label: const Text('Potong / Fokus Daun'),
                        onSelected: _isLoading
                            ? null
                            : (val) => setState(
                                () => _mode = val
                                    ? _InteractionMode.crop
                                    : _InteractionMode.none,
                              ),
                      ),
                    if (_isCropped)
                      TextButton.icon(
                        onPressed: _isLoading ? null : _resetCrop,
                        icon: const Icon(Icons.restore, size: 18),
                        label: const Text('Reset Potongan'),
                      ),
                    ElevatedButton.icon(
                      onPressed: (_imageBytes == null || _isLoading)
                          ? null
                          : _analyzeCorn,
                      icon: const Icon(Icons.document_scanner),
                      label: const Text('Analisa CV'),
                    ),
                    if (_imageBytes != null && _canPreview)
                      FilterChip(
                        selected: _mode == _InteractionMode.gt,
                        selectedColor: Colors.teal.shade100,
                        checkmarkColor: Colors.teal.shade900,
                        avatar: Icon(
                          Icons.crop_free,
                          size: 18,
                          color: Colors.teal.shade800,
                        ),
                        label: const Text('Tandai GT (Hijau)'),
                        onSelected: _isLoading
                            ? null
                            : (val) => setState(
                                () => _mode = val
                                    ? _InteractionMode.gt
                                    : _InteractionMode.none,
                              ),
                      ),
                    if (_gtBoxes.isNotEmpty)
                      ElevatedButton.icon(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.teal.shade700,
                          foregroundColor: Colors.white,
                        ),
                        onPressed: _isLoading ? null : _analyzeGtBoxes,
                        icon: const Icon(Icons.biotech),
                        label: const Text('Analisa Kotak Hijau'),
                      ),
                    if (_gtBoxes.isNotEmpty)
                      TextButton.icon(
                        onPressed: _isLoading
                            ? null
                            : () => setState(() {
                                _gtBoxes.clear();
                                if (_isManualRoiReport) {
                                  _detections = [];
                                  _analysis = null;
                                  _isManualRoiReport = false;
                                }
                              }),
                        icon: const Icon(Icons.delete_outline, size: 18),
                        label: const Text('Hapus GT'),
                      ),
                  ],
                ),
                const SizedBox(height: 20),
                _buildGtEvaluationCard(),
                if (_isLoading)
                  Padding(
                    padding: const EdgeInsets.all(20),
                    child: Column(
                      children: [
                        const CircularProgressIndicator(),
                        const SizedBox(height: 12),
                        Text(_message, textAlign: TextAlign.center),
                      ],
                    ),
                  )
                else if (_analysis != null) ...[
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: _statusColor(status),
                        width: 1.5,
                      ),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 6,
                              ),
                              decoration: BoxDecoration(
                                color: _statusColor(status),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Text(
                                status,
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 13,
                                ),
                              ),
                            ),
                            Text(
                              '${_isManualRoiReport ? 'Mode Manual ROI' : 'Mode Otomatis'} • ${_detections.length} ROI',
                              style: const TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w600,
                                color: Colors.black54,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 14),
                        _buildSectionItem(
                          'Ringkasan Inferensi Computer Vision',
                          (_analysis!['ringkasan_cv'] ?? '-').toString(),
                        ),
                        _buildSectionItem(
                          'Dugaan Penyebab / Hama',
                          '${_analysis!['dugaan_hama'] ?? '-'} (Estimasi area rusak: ${_analysis!['estimasi_area_rusak_persen'] ?? '-'})',
                        ),
                        _buildSectionItem(
                          'Rekomendasi Penanganan',
                          (_analysis!['rekomendasi'] ?? '-').toString(),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey.shade300),
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          'Analisis Ekstraksi Fitur Citra (Feature Extraction)',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 15,
                          ),
                        ),
                        const Divider(height: 20),
                        _buildSectionItem(
                          '1. Analisis Warna & Kontras (Color Space)',
                          (fitur['fitur_warna'] ?? '-').toString(),
                        ),
                        _buildSectionItem(
                          '2. Analisis Morfologi & Tekstur (Shape & Edge)',
                          (fitur['fitur_morfologi'] ?? '-').toString(),
                        ),
                        _buildSectionItem(
                          '3. Verifikasi Kerusakan Hama vs Bercak (Mitigasi False Positive)',
                          (fitur['catatan_false_positive'] ?? '-').toString(),
                        ),
                      ],
                    ),
                  ),
                  if (_detections.isNotEmpty) ...[
                    const SizedBox(height: 16),
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: roiBorder),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _isManualRoiReport
                                ? 'Hasil Diagnosa AI pada Kotak Hijau — ${_detections.length} Area'
                                : 'Detail Region of Interest (ROI) — ${_detections.length} Prediksi',
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 15,
                            ),
                          ),
                          const Divider(height: 20),
                          ...List.generate(_detections.length, (i) {
                            final d = _detections[i];
                            final confPct = (d.confidence * 100)
                                .toStringAsFixed(0);
                            return Container(
                              margin: const EdgeInsets.only(bottom: 10),
                              padding: const EdgeInsets.all(12),
                              decoration: BoxDecoration(
                                color: roiBg,
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Row(
                                    mainAxisAlignment:
                                        MainAxisAlignment.spaceBetween,
                                    children: [
                                      Text(
                                        '${d.isManualGt ? 'Kotak Hijau' : 'Prediksi'} #${i + 1}: ${d.label}',
                                        style: TextStyle(
                                          fontWeight: FontWeight.bold,
                                          color: roiAccent,
                                        ),
                                      ),
                                      Container(
                                        padding: const EdgeInsets.symmetric(
                                          horizontal: 8,
                                          vertical: 2,
                                        ),
                                        decoration: BoxDecoration(
                                          color: roiAccent,
                                          borderRadius: BorderRadius.circular(
                                            10,
                                          ),
                                        ),
                                        child: Text(
                                          'Conf: $confPct%',
                                          style: const TextStyle(
                                            color: Colors.white,
                                            fontSize: 11,
                                            fontWeight: FontWeight.bold,
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                  const SizedBox(height: 4),
                                  Text(
                                    'Bounding Box: ${_formatPixelCoords(d.box)}',
                                    style: const TextStyle(
                                      fontSize: 12,
                                      fontFamily: 'monospace',
                                      color: Colors.black87,
                                    ),
                                  ),
                                  const SizedBox(height: 4),
                                  SelectableText(
                                    'Ciri ROI: ${d.fiturVisual}',
                                    style: const TextStyle(fontSize: 13),
                                  ),
                                ],
                              ),
                            );
                          }),
                        ],
                      ),
                    ),
                  ],
                ] else
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade100,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: SelectableText(_message),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _CvOverlayPainter extends CustomPainter {
  final List<CvDetection> detections;
  final List<NormBox> gtBoxes;
  final NormBox? activeDraft;
  final bool isCropDraft;

  _CvOverlayPainter({
    required this.detections,
    required this.gtBoxes,
    required this.activeDraft,
    required this.isCropDraft,
  });

  Rect _toRect(NormBox b, Size size) {
    return Rect.fromLTRB(
      (b.xmin / 1000) * size.width,
      (b.ymin / 1000) * size.height,
      (b.xmax / 1000) * size.width,
      (b.ymax / 1000) * size.height,
    );
  }

  void _drawTag(
    Canvas canvas,
    Size size,
    Rect rect,
    String text,
    Color bgColor, {
    bool placeAtBottom = false,
  }) {
    final tp = TextPainter(
      text: TextSpan(
        text: ' $text ',
        style: const TextStyle(
          color: Colors.white,
          fontSize: 11,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    final left = rect.left
        .clamp(0.0, math.max(0.0, size.width - tp.width))
        .toDouble();
    final top =
        (placeAtBottom
                ? rect.bottom.clamp(0.0, math.max(0.0, size.height - tp.height))
                : (rect.top - tp.height).clamp(
                    0.0,
                    math.max(0.0, size.height - tp.height),
                  ))
            .toDouble();

    canvas.drawRect(
      Rect.fromLTWH(left, top, tp.width, tp.height),
      Paint()..color = bgColor,
    );
    tp.paint(canvas, Offset(left, top));
  }

  @override
  void paint(Canvas canvas, Size size) {
    // 1. Kotak Hijau (Manual ROI / GT)
    final gtStroke = Paint()
      ..color = Colors.teal.shade600
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.8;
    final gtFill = Paint()
      ..color = const Color.fromRGBO(0, 150, 136, 0.12)
      ..style = PaintingStyle.fill;

    for (var i = 0; i < gtBoxes.length; i++) {
      final rect = _toRect(gtBoxes[i], size);
      canvas.drawRect(rect, gtFill);
      canvas.drawRect(rect, gtStroke);
      _drawTag(
        canvas,
        size,
        rect,
        'GT #${i + 1}',
        Colors.teal.shade700,
        placeAtBottom: true,
      );
    }

    // 2. Kotak Oranye (Prediksi Otomatis AI — tidak digambar ulang jika mode Manual GT)
    final predStroke = Paint()
      ..color = Colors.deepOrange
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.8;
    final predFill = Paint()
      ..color = const Color.fromRGBO(255, 87, 34, 0.14)
      ..style = PaintingStyle.fill;

    for (var i = 0; i < detections.length; i++) {
      final d = detections[i];
      if (d.isManualGt) continue;
      final rect = _toRect(d.box, size);
      final confPct = (d.confidence * 100).round();

      canvas.drawRect(rect, predFill);
      canvas.drawRect(rect, predStroke);
      _drawTag(
        canvas,
        size,
        rect,
        '#${i + 1} ${d.label} $confPct%',
        Colors.deepOrange,
      );
    }

    // 3. Draft kotak saat sedang ditarik mouse
    if (activeDraft != null) {
      final rect = _toRect(activeDraft!, size);
      final fillColor = isCropDraft
          ? const Color.fromRGBO(25, 118, 210, 0.16)
          : const Color.fromRGBO(0, 137, 123, 0.16);
      final strokeColor = isCropDraft
          ? Colors.blue.shade700
          : Colors.teal.shade600;
      canvas.drawRect(
        rect,
        Paint()
          ..color = fillColor
          ..style = PaintingStyle.fill,
      );
      canvas.drawRect(
        rect,
        Paint()
          ..color = strokeColor
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2.5,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _CvOverlayPainter oldDelegate) {
    return true;
  }
}
