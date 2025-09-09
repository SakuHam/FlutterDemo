import 'package:flutter/material.dart';
import 'game.dart';
import 'app_grid.dart';
import 'music.dart';

void main() {
  runApp(const MoonLanderApp());
}

class MoonLanderApp extends StatelessWidget {
  const MoonLanderApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Moon Lander',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0B0F14),
        textTheme: const TextTheme(bodyMedium: TextStyle(fontFamily: 'monospace')),
      ),
      home: const MainPage(),
    );
  }
}

class MainPage extends StatefulWidget {
  const MainPage({super.key});
  @override
  State<MainPage> createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  late List<AppIcon> icons;

  @override
  void initState() {
    super.initState();
    icons = [
      AppIcon(
        id: 'game',
        label: 'Moon Lander',
        icon: Icons.rocket_launch,
        row: 0,
        col: 0,
        onTap: () {
          Navigator.push(context, MaterialPageRoute(builder: (_) => const GamePage()));
        },
      ),
      AppIcon(
        id: 'music',
        label: 'Music',
        icon: Icons.music_note,
        row: 0,
        col: 1,
        onTap: () {
          Navigator.push(context, MaterialPageRoute(builder: (_) => const MusicPage()));
        },
      ),
      AppIcon(
        id: 'settings',
        label: 'Settings',
        icon: Icons.settings,
        row: 0,
        col: 2,
        onTap: null, // placeholder
      ),
    ];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: AppIconGrid(
          icons: icons,
          cellSize: 96,
          gap: 16,
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
          header: const Text('Apps', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold)),
          onChanged: (updated) {
            // If you want to persist positions, save `updated` somewhere.
            setState(() {});
          },
        ),
      ),
    );
  }
}
