from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton

class ThresholdDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Set Threshold Values")

        layout = QVBoxLayout()
        self.red_input = QLineEdit(self)
        self.green_input = QLineEdit(self)
        self.blue_input = QLineEdit(self)

        layout.addWidget(QLabel("Red Threshold:"))
        layout.addWidget(self.red_input)
        layout.addWidget(QLabel("Green Threshold:"))
        layout.addWidget(self.green_input)
        layout.addWidget(QLabel("Blue Threshold:"))
        layout.addWidget(self.blue_input)

        submit_button = QPushButton("Apply", self)
        submit_button.clicked.connect(self.accept)
        layout.addWidget(submit_button)

        self.setLayout(layout)

    def get_values(self):
        try:
            r = int(self.red_input.text())
            g = int(self.green_input.text())
            b = int(self.blue_input.text())
            return r, g, b
        except ValueError:
            return None