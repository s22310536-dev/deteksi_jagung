import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:deteksi_jagung/main.dart';

void main() {
  testWidgets('Halaman Computer Vision deteksi hama jagung tampil', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const MaterialApp(home: CornVisionPage()));

    expect(find.text('Computer Vision — Deteksi Hama Jagung'), findsOneWidget);
    expect(find.text('Pilih Gambar'), findsOneWidget);
    expect(find.text('Analisa CV'), findsOneWidget);
  });
}
