import sys
import pandas as pd
import pickle
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox, QLineEdit, QFileDialog, QMessageBox)
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import QFont 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model_pkl_file = "DecisionTreeModel.pkl"

class DecisionTreeApp(QWidget):

    def __init__(self):
        super().__init__()

        # Variabel untuk menyimpan Decision Tree model
        self.dtree = None
        self.df = None

        #load model from pickle file
        try:
            with open(model_pkl_file, 'rb') as file:
                self.dtree = pickle.load(file)
            with open('X_test.pkl', 'rb') as file:
                X_test = pickle.load(file)
            with open('y_test.pkl', 'rb') as file:
                y_test = pickle.load(file)
            y_predict = self.dtree.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_predict)*100
        except FileNotFoundError:
            print("Belum ada model pelatihan yang disimpan silahkan upload file terlebih dahulu")

        # Layout utama
        self.layout = QVBoxLayout()

        # Set ikon jendela
        self.setWindowIcon(QIcon('icon/unsri.png'))

        # Label Judul
        self.title_label = QLabel('Klasifikasi Data Jasa Service PT. Gobel Dharma Nusantara', self)
        font = QFont()
        font.setBold(True) 
        font.setPointSize(12)
        self.title_label.setFont(font)
        self.layout.addWidget(self.title_label)

        # Dropdown ASCCode
        self.asc_code_label = QLabel('Pilih Service Center:')
        self.asc_code_combo = QComboBox()
        asc_code_options = sorted({
            '2501 BALIKPAPAN': 2501, '5601 MATARAM': 5601, '6101 RANTAU PRAPAT': 6101, 
            '2301 TANJUNG KARANG': 2301, '5301 JAKARTA 6': 5301, '3201 PONTIANAK': 3201, 
            '3601 MALANG': 3601, '4901 JEMBER': 4901, '2001 MEDAN': 2001, '4101 BATAM': 4101, 
            '6701 PINRANG SERVICE': 6701, '1701 BANDUNG': 1701, '1401 JAK KOTA': 1401, 
            '4601 SAMARINDA': 4601, '6801 LHOKSEUMAWE': 6801, '6501 PEKALONGAN': 6501, 
            '1201 JAKARTA SELATAN I': 1201, '5801 AMBON': 5801, '4501 PURWOKERTO': 4501, 
            '3901 SURABAYA II': 3901, '5201 KEDIRI': 5201, '5401 PANGKAL PINANG': 5401, 
            '2801 BANDA ACEH': 2801, '5701 TASIKMALAYA': 5701, '2401 BANJARMASIN': 2401, 
            '3401 JAMBI': 3401, '2601 MAKASSAR': 2601, '1901 SURABAYA I': 1901, '2101 PALEMBANG': 2101,
            '3701 JAKARTA BSD': 3701, '5501 SAMPIT': 5501, '6601 SUKABUMI': 6601, 
            '2201 PEKANBARU': 2201, '6201 BENGKULU': 6201, '4801 GORONTALO': 4801, 
            '3001 YOGYAKARTA': 3001, '3301 PALU': 3301, '1801 SEMARANG': 1801, 
            '4401 SOLO': 4401, '4201 BOGOR': 4201, '4701 KENDARI': 4701, 
            '6401 JAYAPURA': 6401, '3501 CIREBON': 3501, '6001 CILEGON': 6001, 
            '2901 PADANG': 2901, '2701 MANADO': 2701, '6301 MUARA BUNGO': 6301, 
            '5901 TARAKAN': 5901, '4301 BEKASI': 4301, '3801 JAKARTA KELAPA GADING': 3801, 
            '3101 DENPASAR': 3101, '4001 SURABAYA III': 4001, '1301 JAKARTA SELATAN II': 1301,
        }.items())
        for option, value in asc_code_options:
            self.asc_code_combo.addItem(option, value)
        self.layout.addWidget(self.asc_code_label)
        self.layout.addWidget(self.asc_code_combo)

        # Dropdown MQC
        self.mqc_label = QLabel('Minimum Quality Control (1 to 6):')
        self.mqc_combo = QComboBox()
        self.mqc_combo.addItems([str(i) for i in range(1, 7)])
        self.layout.addWidget(self.mqc_label)
        self.layout.addWidget(self.mqc_combo)

        # Input untuk Service Repair, Service Trans, Part Repair
        self.service_repair_label = QLabel('Biaya Service Repair:')
        self.service_repair_input = QLineEdit()
        self.layout.addWidget(self.service_repair_label)
        self.layout.addWidget(self.service_repair_input)

        self.service_trans_label = QLabel('Biaya Service Trans:')
        self.service_trans_input = QLineEdit()
        self.layout.addWidget(self.service_trans_label)
        self.layout.addWidget(self.service_trans_input)

        self.part_repair_label = QLabel('Biaya Part Repair:')
        self.part_repair_input = QLineEdit()
        self.layout.addWidget(self.part_repair_label)
        self.layout.addWidget(self.part_repair_input)

        # Dropdown In Warranty
        self.in_warranty_label = QLabel('Dalam Masa Garansi:')
        self.in_warranty_combo = QComboBox()
        self.in_warranty_combo.addItems(['Tidak', 'Ya'])
        self.layout.addWidget(self.in_warranty_label)
        self.layout.addWidget(self.in_warranty_combo)

        # Dropdown Remarks
        self.remarks_label = QLabel('Remarks/ Keterangan Tambahan:')
        self.remarks_combo = QComboBox()
        self.remarks_combo.addItems(['No Sparepart', 'Use Own Stock', 'Order Part to CPC'])
        self.layout.addWidget(self.remarks_label)
        self.layout.addWidget(self.remarks_combo)

        # Dropdown Respon
        self.respon_label = QLabel('Waktu Respon/Tindakan yang diambil:')
        self.respon_combo = QComboBox()
        self.respon_combo.addItems(['1 Days', '>1 Days'])
        self.layout.addWidget(self.respon_label)
        self.layout.addWidget(self.respon_combo)

        # Dropdown Chategory
        self.chategory_label = QLabel('Kategori:')
        self.chategory_combo = QComboBox()
        self.chategory_combo.addItems(['AIR CONDITIONER', 'CRT TV', 'DSC / DVC', 'DVD / PLAYER',
                                       'GENERAL AUDIO', 'LED / LCD', 'PDP', 'REFRIGERATOR', 
                                       'SMALL HAPP', 'SYSTEM SOLUTION', 'VRF', 'WASHING MACHINE'])
        self.layout.addWidget(self.chategory_label)
        self.layout.addWidget(self.chategory_combo)

        # Tombol untuk upload file CSV
        self.upload_button = QPushButton('Latih dan Simpan Model')
        self.upload_button.clicked.connect(self.upload_csv)
        self.layout.addWidget(self.upload_button)

        # Tombol untuk melakukan prediksi
        self.predict_button = QPushButton('Prediksi')
        self.predict_button.clicked.connect(self.predict)
        self.layout.addWidget(self.predict_button)

        # Set layout ke window
        self.setLayout(self.layout)

        # Set geometry
        self.setGeometry(100, 100, 100, 100)

    def upload_csv(self):
        # Fungsi untuk mengunggah dan memproses file CSV
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih data file CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_path:
            if file_path.endswith('.csv'):
                try:
                    self.df = pd.read_csv(file_path)
                    
                    # 1. Bersihkan nama kolom dari spasi
                    self.df.columns = self.df.columns.str.strip()
                    
                    # 2. Cek apakah DataFrame kosong
                    if self.df.empty:
                        QMessageBox.warning(self, 'File Upload', 'File CSV kosong atau tidak memiliki header.')
                        return  # Berhenti jika DataFrame kosong
                    
                    # 3. Cek keberadaan kolom
                    required_columns = ['Ppp ASCCode']  # Kolom yang harus ada
                    missing_columns = [col for col in required_columns if col not in self.df.columns]
                    
                    if missing_columns:
                        QMessageBox.warning(self, 'File Upload', f"Kolom berikut tidak ditemukan di file CSV: {', '.join(missing_columns)}")
                        return  # Berhenti jika kolom tidak ditemukan
                    
                    # Jika semua berjalan lancar, lanjutkan proses
                    self.train_model()
                except Exception as e:
                    QMessageBox.warning(self, 'File Upload', f"Terjadi kesalahan saat memuat file CSV: {e}")
            else:
                QMessageBox.warning(self, 'File Upload', 'Jenis file tidak didukung. Harap unggah file dengan format .csv.')
        else:
            QMessageBox.warning(self, 'File Upload', 'Tidak ada file yang dipilih')

    def train_model(self):
        if self.df is not None:
            # Lakukan mapping kolom-kolom seperti di contoh sebelumnya
            d_remarks = {'No Sparepart': 0, 'Use Own Stock': 1, 'Order Part to CPC': 2}
            self.df['Remarks'] = self.df['Remarks'].map(d_remarks)

            d_respon = {'1 Days': 0, '>1 Days': 1}
            self.df['RESPON'] = self.df['RESPON'].map(d_respon)

            d_chategory = {
                'AIR CONDITIONER': 0, 'CRT TV': 1, 'DSC / DVC': 2, 'DVD / PLAYER': 3, 
                'GENERAL AUDIO': 4, 'LED / LCD': 5, 'PDP': 6, 'REFRIGERATOR': 7, 
                'SMALL HAPP': 8, 'SYSTEM SOLUTION': 9, 'VRF': 10, 'WASHING MACHINE': 11
            }
            self.df['Chategory'] = self.df['Chategory'].map(d_chategory)

            d_speed = {'2 Days': 0, '>2 Days': 1}
            self.df['SPEED'] = self.df['SPEED'].map(d_speed)

            features = ['Ppp ASCCode', 'ppp mqc', 'Service Repair', 'Service Trans', 'Part Repair', 
                        'In Warranty', 'RESPON', 'Remarks', 'Chategory']

            X = self.df[features]
            y = self.df['SPEED']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
            
            # Membangun Decision Tree
            self.dtree = DecisionTreeClassifier()
            self.dtree = self.dtree.fit(X_train, y_train)
            y_predict = self.dtree.predict(X_test)

            model_pkl_file = "DecisionTreeModel.pkl"
            with open('X_test.pkl', 'wb') as file:
                pickle.dump(X_test, file)
            with open('y_test.pkl', 'wb') as file:
                    pickle.dump(y_test, file)          
            with open(model_pkl_file, 'wb') as file:
                pickle.dump(self.dtree, file)

            # Evaluasi model dengan ten-fold cross-validation
            cv_scores = cross_val_score(self.dtree, X, y, cv=10)
            self.accuracy = accuracy_score(y_test, y_predict)*100
            report = classification_report(y_test, y_predict)  
            confusion_m = confusion_matrix(y_test, y_predict)

            QMessageBox.information(self, 'Success Train Model!', 'Model berhasil dilatih dan disimpan\n'
                                    f"Accuracy: {self.accuracy:.2f}%\nClassification Report:\n{report}\n"
                                    f"Confusion Matrix:\n{confusion_m}\n"
                                    f"10-Fold Cross-Validation Scores: {cv_scores}\n"
                                    f"Mean CV Score: {cv_scores.mean()}")

    def predict(self):
        if self.dtree is None:
            QMessageBox.warning(self, 'Error', 'Upload terlebih dahulu data file CSV')
            return
        
        # Ambil input dari pengguna
        asc_code = self.asc_code_combo.currentData()
        mqc = int(self.mqc_combo.currentText())
        service_repair = self.service_repair_input.text()
        service_trans = self.service_trans_input.text()
        part_repair = self.part_repair_input.text()

        # Validasi input Service Repair, Service Trans, Part Repair
        if not (service_repair.isdigit() and service_trans.isdigit() and part_repair.isdigit()):
            QMessageBox.warning(self, 'Input Error', 'Biaya Service Repair, Biaya Service Trans, dan Biaya Part Repair harus diisi angka')
            return

        in_warranty = 1 if self.in_warranty_combo.currentText() == 'Ya' else 0
        respon = 1 if self.respon_combo.currentText() == '>1 Days' else 0
        remarks = self.remarks_combo.currentIndex()
        chategory = self.chategory_combo.currentIndex()

        # Lakukan prediksi berdasarkan input
        input_data = [[asc_code, mqc, int(service_repair), int(service_trans), int(part_repair), in_warranty, respon, remarks, chategory]]
        prediction = self.dtree.predict(input_data)
        if prediction == 0:
            QMessageBox.information(self, 'Hasil!', f'Waktu yang dibutuhkan untuk tindakan jasa service dalam dua hari\n'
                                    f"Dengan tingkat akurasi prediksi {self.accuracy:.2f}%")
        else:
            QMessageBox.information(self, 'Hasil!', f'Waktu yang dibutuhkan untuk tindakan jasa service lebih dari dua hari\n'
                                    f"Dengan tingkat akurasi prediksi {self.accuracy:.2f}%")
        
# Menjalankan aplikasi
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DecisionTreeApp()
    window.setWindowTitle('Klasifikasi Data Jasa Service Menggunakan Decision Tree')
    window.setGeometry(100, 100, 100, 100)  # Set geometry window
    window.show()
    sys.exit(app.exec_())
