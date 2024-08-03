import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QFileDialog, QLabel, QVBoxLayout, QWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl
import torch
import torchaudio
from speechbrain.inference.enhancement import SpectralMaskEnhancement

class SpeechEnhancementApp(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set up the model
        self.enhance_model = SpectralMaskEnhancement.from_hparams(
            source="speechbrain/metricgan-plus-voicebank",
            savedir="pretrained_models/metricgan-plus-voicebank",
        )

        # Set up the UI
        self.setWindowTitle("Speech Enhancement")
        self.setGeometry(100, 100, 400, 200)

        self.layout = QVBoxLayout()

        self.load_button = QPushButton("Load Noisy Audio", self)
        self.load_button.clicked.connect(self.load_audio)
        self.layout.addWidget(self.load_button)

        self.enhance_button = QPushButton("Enhance Audio", self)
        self.enhance_button.clicked.connect(self.enhance_audio)
        self.layout.addWidget(self.enhance_button)

        self.play_button = QPushButton("Play Enhanced Audio", self)
        self.play_button.setDisabled(True)  # Initially disabled
        self.play_button.clicked.connect(self.play_audio)
        self.layout.addWidget(self.play_button)

        self.status_label = QLabel("", self)
        self.layout.addWidget(self.status_label)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

        self.noisy_audio_path = ""
        self.enhanced_audio_path = "enhanced.wav"

        # Set up the media player
        self.media_player = QMediaPlayer(None, QMediaPlayer.LowLatency)

    def load_audio(self):
        # Open file dialog to select a noisy audio file
        file_dialog = QFileDialog()
        audio_path, _ = file_dialog.getOpenFileName(self, "Open Noisy Audio", "", "Audio Files (*.wav)")
        if audio_path:
            self.noisy_audio_path = audio_path
            self.status_label.setText(f"Loaded: {audio_path}")

    def enhance_audio(self):
        if self.noisy_audio_path:
            self.status_label.setText("Enhancing...")
            noisy = self.enhance_model.load_audio(self.noisy_audio_path).unsqueeze(0)
            enhanced = self.enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
            torchaudio.save(self.enhanced_audio_path, enhanced.cpu(), 16000)
            self.status_label.setText(f"Enhanced audio saved as '{self.enhanced_audio_path}'")
            self.play_button.setEnabled(True)  # Enable the play button
        else:
            self.status_label.setText("No audio file loaded!")

    def play_audio(self):
        url = QUrl.fromLocalFile(self.enhanced_audio_path)
        self.media_player.setMedia(QMediaContent(url))
        self.media_player.play()
        self.status_label.setText("Playing enhanced audio...")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SpeechEnhancementApp()
    window.show()
    sys.exit(app.exec_())
