import tkinter as tk
from tkinter import ttk, filedialog
from tkinter.filedialog import askopenfilename
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tkinter import ttk, messagebox
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class AplikasiSaham(tk.Tk):
    def __init__(self):
        super().__init__()
        self.prediksi_terakhir = []
        self.title("Aplikasi Prediksi Saham dan Risiko Investasi")
        self.geometry("600x400")

        # Frame Halaman
        self.halaman_utama = HalamanUtama(self)
        self.halaman_unggah = HalamanUnggah(self)
        self.halaman_prediksi = HalamanPrediksi(self)
        self.halaman_risiko = HalamanRisiko(self)

        self.halaman_utama.pack(fill="both", expand=True)

    def buka_halaman_unggah(self):
        self.halaman_utama.pack_forget()
        self.halaman_prediksi.pack_forget()
        self.halaman_risiko.pack_forget()
        self.halaman_unggah.pack(fill="both", expand=True)

    def kembali_ke_halaman_utama(self):
        self.halaman_unggah.pack_forget()
        self.halaman_prediksi.pack_forget()
        self.halaman_risiko.pack_forget()
        self.halaman_utama.pack(fill="both", expand=True)

    def buka_halaman_prediksi(self):
        self.halaman_unggah.pack_forget()
        self.halaman_utama.pack_forget()
        self.halaman_risiko.pack_forget()
        self.halaman_prediksi.pack(fill="both", expand=True)
    
    def buka_halaman_risiko(self):
        self.halaman_utama.pack_forget()
        self.halaman_unggah.pack_forget()
        self.halaman_prediksi.pack_forget()
        self.halaman_risiko.pack(fill="both", expand=True)

class HalamanUtama(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # Judul Halaman Utama
        tk.Label(
            self, text="Aplikasi Prediksi Saham dan Risiko Investasi", 
            font=("Arial", 16, "bold")
        ).pack(pady=20)

        # Placeholder frame untuk gambar atau teks proyek
        frame_gambar = tk.Frame(self, width=300, height=200, bg="lightgray")
        frame_gambar.pack(pady=10)

        teks_proyek = tk.Label(
            frame_gambar, 
            text="Proyek ini memprediksi harga dan\nrisiko saham blue chip seperti MYOR & TBIG.\n"
                 "Dengan model GRU, kami memberikan\nanalisis yang akurat untuk investasi.",
            font=("Arial", 12), 
            bg="lightgray", 
            justify="center"
        )
        teks_proyek.place(relx=0.5, rely=0.5, anchor="center")

        # Frame for side-by-side buttons
        button_frame = tk.Frame(self)
        button_frame.pack(pady=20)

        # Tombol untuk menuju halaman unggah data dan prediksi
        btn_unggah = ttk.Button(button_frame, text="Lihat Grafik", 
                                command=self.parent.buka_halaman_unggah)
        btn_unggah.pack(side="left", padx=10)
        
        btn_prediksi = ttk.Button(button_frame, text="Halaman Prediksi", 
                                  command=self.parent.buka_halaman_prediksi)
        btn_prediksi.pack(side="left", padx=10)
        
        btn_var = ttk.Button(button_frame, text="Halaman VaR", 
                             command=self.parent.buka_halaman_risiko)
        btn_var.pack(side="left", padx=10)

class HalamanUnggah(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.file_path = None
        self.data_ready = False # Variabel untuk menyimpan path file yang diunggah

        # Judul Halaman
        tk.Label(self, text="Unggah Data dan Lihat Statistik", font=("Arial", 14, "bold")).pack(pady=10)

        # Tombol Unggah File
        btn_unggah_file = ttk.Button(self, text="Unggah Data", command=self.unggah_file)
        btn_unggah_file.place(x=10, y=20)

        # Tombol Update Graph
        btn_update = ttk.Button(self, text="Update Graph", command=self.update_graph)
        btn_update.place(x=120, y=20)

        btn_windows_data = ttk.Button(self, text="Windows Data", command=self.process_data)
        btn_windows_data.place(x=220, y=20)

        # Placeholder untuk Grafik
        self.canvas_frame = tk.Frame(self, width=500, height=300, bg="white", relief="solid", borderwidth=1)
        self.canvas_frame.place(relx=0.5, rely=0.4, anchor="center")

        # Placeholder untuk Statistik Deskriptif
        self.stats_frame = tk.Frame(self, width=500, height=150, bg="white", relief="solid", borderwidth=1)
        self.stats_frame.place(relx=0.5, rely=0.8, anchor="center")

        # Tombol Back ke Halaman Utama
        btn_back = ttk.Button(self, text="Back", command=self.parent.kembali_ke_halaman_utama)
        btn_back.place(relx=0.85, rely=0.9)

    def unggah_file(self):
        # Dialog untuk memilih file
        self.file_path = filedialog.askopenfilename(
            title="Pilih File Data Saham",
            filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
        )
        if self.file_path:
            print(f"File dipilih: {self.file_path}")

    def update_graph(self):
        # Pastikan file telah dipilih
        if not self.file_path:
            tk.messagebox.showerror("Error", "Silakan unggah file terlebih dahulu.")
            return

        # Membaca file CSV
        try:
            data = pd.read_csv(self.file_path)

            # Validasi kolom data
            if 'Date' not in data.columns or 'Close' not in data.columns:
                tk.messagebox.showerror("Error", "File harus memiliki kolom 'Date' dan 'Close'.")
                return

            # Proses data
            data['Date'] = pd.to_datetime(data['Date'])
            data.sort_values('Date', inplace=True)

            # Bersihkan frame untuk grafik dan statistik
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()
            for widget in self.stats_frame.winfo_children():
                widget.destroy()

            # Membuat grafik
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(data['Date'], data['Close'], label="Close Price", color='blue')
            ax.set_title("Closing Prices Over Time", fontsize=14)
            ax.set_xlabel("Date", fontsize=12)
            ax.set_ylabel("Price", fontsize=12)
            ax.legend()

            # Menampilkan grafik di tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill="both", expand=True)
            canvas.draw()

            # Menampilkan statistik deskriptif
            stats = data['Close'].describe()
            stats_text = (
                f"Statistika Deskriptif:\n"
                f"- Count: {stats['count']}\n"
                f"- Mean: {stats['mean']:.2f}\n"
                f"- Std: {stats['std']:.2f}\n"
                f"- Min: {stats['min']:.2f}\n"
                f"- 25%: {stats['25%']:.2f}\n"
                f"- 50% (Median): {stats['50%']:.2f}\n"
                f"- 75%: {stats['75%']:.2f}\n"
                f"- Max: {stats['max']:.2f}"
            )
            stats_label = tk.Label(self.stats_frame, text=stats_text, font=("Arial", 12), justify="left", bg="white")
            stats_label.pack(anchor="w", padx=10, pady=10)
        except Exception as e:
            tk.messagebox.showerror("Error", f"Terjadi kesalahan saat memproses file:\n{e}")
    def process_data(self):
        if not self.file_path:
            tk.messagebox.showerror("Error", "Silakan unggah file terlebih dahulu.")
            return
        try:
            # Baca data
            data = pd.read_csv(self.file_path)

            # Validasi kolom
            if 'Close' not in data.columns:
                tk.messagebox.showerror("Error", "File harus memiliki kolom 'Close'.")
                return

            # Data Preparation
            features = ['Close']
            df = data[features]
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df)

            def create_sequences(data, window_size=6):
                X, y = [], []
                for i in range(len(data) - window_size):
                    X.append(data[i:i + window_size])
                    y.append(data[i + window_size][0])
                return np.array(X), np.array(y)

            # Sequence length
            time_steps = 40
            X, y = create_sequences(scaled_data, time_steps)

            # Split data
            split = int(0.8 * len(X))
            self.X_train, self.X_test = X[:split], X[split:]
            self.y_train, self.y_test = y[:split], y[split:]

            self.data_ready = True
            tk.messagebox.showinfo("Sukses", "Data berhasil diproses!")

        except Exception as e:
            tk.messagebox.showerror("Error", f"Terjadi kesalahan saat memproses data:\n{e}")   
        

class HalamanPrediksi(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.model = None  # Model yang akan dibuat
        self.layer_config = []  # Daftar layer yang dipilih
        
        # Canvas dan Scrollbar
        canvas_frame = tk.Frame(self)
        canvas_frame.pack(fill="both", expand=True)

        canvas = tk.Canvas(canvas_frame)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Frame dalam Canvas
        content_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=content_frame, anchor="nw")
        content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # Layout Elemen UI
        tk.Label(content_frame, text="Konfigurasi Dinamis Model Prediksi", font=("Arial", 16, "bold")).pack(pady=10)

        ttk.Button(content_frame, text="Cek Data", command=self.check_data).pack(pady=10)

        # Pilihan Layer
        tk.Label(content_frame, text="Pilih Layer:", font=("Arial", 12)).pack(anchor="w", padx=20)
        self.layer_choice = ttk.Combobox(content_frame, values=["GRU", "Dropout", "Dense"], state="readonly")
        self.layer_choice.pack(anchor="w", padx=20)
        ttk.Button(content_frame, text="Konfigurasi Layer", command=self.konfigurasi_layer).pack(anchor="w", padx=20, pady=5)

        # Preview Layer
        tk.Label(content_frame, text="Arsitektur Model:", font=("Arial", 12)).pack(anchor="w", padx=20)
        self.layer_preview = tk.Text(content_frame, width=60, height=10, state="disabled", wrap="word")
        self.layer_preview.pack(anchor="w", padx=20, pady=10)

        # Pilihan Optimizer
        tk.Label(content_frame, text="Pilih Optimizer:", font=("Arial", 12)).pack(anchor="w", padx=20)
        self.optimizer_choice = ttk.Combobox(content_frame, values=["adam", "sgd", "rmsprop"], state="readonly")
        self.optimizer_choice.pack(anchor="w", padx=20)

        ttk.Button(content_frame, text="Buat Model", command=self.buat_model).pack(pady=10)
        ttk.Button(content_frame, text="Train Model", command=self.train_model).pack(pady=10)

        # Prediksi Data Baru
        prediksi_frame = tk.LabelFrame(content_frame, text="Prediksi Data Baru", padx=10, pady=10)
        prediksi_frame.pack(pady=20, fill="both", expand=True)

        tk.Label(prediksi_frame, text="Jumlah langkah prediksi:").grid(row=0, column=0, sticky="w")
        self.input_steps = tk.Entry(prediksi_frame, width=10)
        self.input_steps.grid(row=0, column=1, padx=10, sticky="w")

        ttk.Button(prediksi_frame, text="Prediksi", command=self.predict_future).grid(row=1, column=0, columnspan=2, pady=10)

        self.prediksi_output = tk.Text(prediksi_frame, height=10, width=50, state="disabled")
        self.prediksi_output.grid(row=2, column=0, columnspan=2, pady=10)

        ttk.Button(content_frame, text="Back", command=self.parent.kembali_ke_halaman_utama).pack(pady=10)

    def check_data(self):
        messagebox.showinfo("Data Check", "Fitur ini belum dikonfigurasi sepenuhnya.")

    def konfigurasi_layer(self):
        layer_type = self.layer_choice.get()
        if not layer_type:
            messagebox.showerror("Error", "Pilih jenis layer terlebih dahulu.")
            return
        if layer_type == "GRU":
            self.konfigurasi_gru()
        elif layer_type == "Dropout":
            self.konfigurasi_dropout()
        elif layer_type == "Dense":
            self.konfigurasi_dense()

    def konfigurasi_gru(self):
        def tambahkan_gru():
            try:
                units = int(entry_units.get())
                return_sequences = var_return_sequences.get()
                self.layer_config.append({"type": "GRU", "units": units, "return_sequences": return_sequences})
                self.update_preview()
                window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Masukkan jumlah unit GRU yang valid.")

        window = tk.Toplevel(self)
        window.title("Konfigurasi GRU")
        tk.Label(window, text="Jumlah Unit:").pack(pady=5)
        entry_units = tk.Entry(window, width=10)
        entry_units.pack(pady=5)
        var_return_sequences = tk.BooleanVar(value=False)
        ttk.Checkbutton(window, text="Return Sequences", variable=var_return_sequences).pack(pady=5)
        ttk.Button(window, text="Tambahkan", command=tambahkan_gru).pack(pady=10)

    def konfigurasi_dropout(self):
        def tambahkan_dropout():
            try:
                rate = float(entry_rate.get())
                if not (0 <= rate <= 1):
                    raise ValueError
                self.layer_config.append({"type": "Dropout", "rate": rate})
                self.update_preview()
                window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Masukkan dropout rate yang valid (0-1).")

        window = tk.Toplevel(self)
        window.title("Konfigurasi Dropout")
        tk.Label(window, text="Dropout Rate (0-1):").pack(pady=5)
        entry_rate = tk.Entry(window, width=10)
        entry_rate.pack(pady=5)
        ttk.Button(window, text="Tambahkan", command=tambahkan_dropout).pack(pady=10)

    def konfigurasi_dense(self):
        def tambahkan_dense():
            try:
                units = int(entry_units.get())
                self.layer_config.append({"type": "Dense", "units": units})
                self.update_preview()
                window.destroy()
            except ValueError:
                messagebox.showerror("Error", "Masukkan jumlah unit Dense yang valid.")

        window = tk.Toplevel(self)
        window.title("Konfigurasi Dense")
        tk.Label(window, text="Jumlah Unit:").pack(pady=5)
        entry_units = tk.Entry(window, width=10)
        entry_units.pack(pady=5)
        ttk.Button(window, text="Tambahkan", command=tambahkan_dense).pack(pady=10)

    def update_preview(self):
        self.layer_preview.config(state="normal")
        self.layer_preview.delete("1.0", "end")
        for i, layer in enumerate(self.layer_config):
            if layer["type"] == "GRU":
                text = f"{i + 1}. GRU(units={layer['units']}, return_sequences={layer['return_sequences']})\n"
            elif layer["type"] == "Dropout":
                text = f"{i + 1}. Dropout(rate={layer['rate']})\n"
            elif layer["type"] == "Dense":
                text = f"{i + 1}. Dense(units={layer['units']})\n"
            self.layer_preview.insert("end", text)
        self.layer_preview.config(state="disabled")

    def tambahkan_layer_output(self):
        """
        Tambahkan layer terakhir ke model (Dense dengan 1 unit dan linear activation).
        """
        self.model.add(Dense(units=1, activation='linear'))

    def buat_model(self):
        if not self.layer_config:
            messagebox.showerror("Error", "Tambahkan setidaknya satu layer.")
            return

        optimizer_name = self.optimizer_choice.get()
        if optimizer_name not in ["adam", "sgd", "rmsprop"]:
            tk.messagebox.showerror("Error", "Pilih optimizer yang valid.")
            return

        try:
            self.model = Sequential()

            # Tambahkan semua layer yang dikonfigurasi
            for layer in self.layer_config:
                if layer["type"] == "GRU":
                    self.model.add(GRU(units=layer["units"], return_sequences=layer["return_sequences"], input_shape=(40, 1)))
                elif layer["type"] == "Dropout":
                    self.model.add(Dropout(rate=layer["rate"]))
                elif layer["type"] == "Dense":
                    self.model.add(Dense(units=layer["units"], activation='relu'))

            # Tambahkan layer output secara eksplisit
            self.tambahkan_layer_output()

            # Pilih dan kompilasi optimizer
            optimizer = {"adam": Adam(), "sgd": SGD(), "rmsprop": RMSprop()}[optimizer_name]
            self.model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])

            tk.messagebox.showinfo("Sukses", "Model berhasil dibuat dan dikompilasi!")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Terjadi kesalahan: {e}")

    def train_model(self):
        if self.model is None:
            tk.messagebox.showerror("Error", "Model belum dibuat.")
            return

        if not hasattr(self.parent.halaman_unggah, 'data_ready') or not self.parent.halaman_unggah.data_ready:
            tk.messagebox.showerror("Error", "Data belum siap! Silakan proses data di halaman unggah.")
            return

        X_train = self.parent.halaman_unggah.X_train
        y_train = self.parent.halaman_unggah.y_train

        self.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        tk.messagebox.showinfo("Sukses", "Model berhasil dilatih!")

    def predict_future(self):
        if self.model is None:
            tk.messagebox.showerror("Error", "Model belum dibuat atau dilatih.")
            return

        try:
            steps = int(self.input_steps.get())
            if steps <= 0:
                raise ValueError

            test_data = self.parent.halaman_unggah.X_test
            x_input2 = test_data[-1:]

            predicted_values = []
            for _ in range(steps):
                prediction = self.model.predict(x_input2)
                predicted_value = prediction[0][0]

                new_sequence = np.copy(x_input2)
                new_sequence[:, :-1] = x_input2[:, 1:]
                new_sequence[:, -1] = predicted_value

                predicted_values.append(predicted_value)
                x_input2 = new_sequence

            # Simpan hasil prediksi ke dalam atribut aplikasi
            self.parent.prediksi_terakhir = predicted_values
            
            self.prediksi_output.config(state="normal")
            self.prediksi_output.delete("1.0", "end")
            self.prediksi_output.insert("end", f"Hasil prediksi ({steps} langkah):\n")
            self.prediksi_output.insert("end", "\n".join([f"Step {i+1}: {val:.4f}" for i, val in enumerate(predicted_values)]))
            self.prediksi_output.config(state="disabled")

        except ValueError:
            tk.messagebox.showerror("Error", "Masukkan jumlah langkah prediksi yang valid.")


class HalamanRisiko(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # Placeholder Grafik Risiko (Matplotlib Figure)
        self.fig, self.ax = plt.subplots(figsize=(4, 3))
        self.ax.set_title("Histogram Returns")
        self.ax.set_xlabel("Returns")
        self.ax.set_ylabel("Frequency")

        # Matplotlib Canvas in Tkinter Frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().place(relx=0.5, rely=0.3, anchor="center")

        # Notifikasi Label
        self.notifikasi = tk.Label(self, text="Notifikasi: Data prediksi belum tersedia untuk menghitung nilai kerugian.", 
                                   font=("Arial", 10), fg="red")
        self.notifikasi.place(relx=0.5, rely=0.55, anchor="center")

        # Frame for buttons
        button_frame = tk.Frame(self)
        button_frame.place(relx=0.5, rely=0.7, anchor="center")

        btn_get_test_data = ttk.Button(button_frame, text="Ambil Data Uji", command=self.ambil_data_uji)
        btn_get_test_data.pack(side="left", padx=10)

        # Button to calculate VaR
        btn_var = ttk.Button(button_frame, text="Hitung VaR", command=self.hitung_var)
        btn_var.pack(side="left", padx=10)

        # Button to return to main page
        btn_back = ttk.Button(button_frame, text="Kembali", command=self.parent.kembali_ke_halaman_utama)
        btn_back.pack(side="left", padx=10)

        # Hasil VaR
        tk.Label(self, text="Hasil Value at Risk:", font=("Arial", 12)).place(relx=0.5, rely=0.8, anchor="center")
        self.hasil_var = tk.Text(self, height=3, width=40, state="disabled")
        self.hasil_var.place(relx=0.5, rely=0.85, anchor="center")
    def ambil_data_uji(self):
        # Pastikan file sudah diunggah
        if not hasattr(self.parent.halaman_unggah, 'file_path') or not self.parent.halaman_unggah.file_path:
            tk.messagebox.showerror("Error", "Silakan unggah file data terlebih dahulu di halaman unggah.")
            return

        try:
            # Baca file CSV dari halaman unggah
            file_path = self.parent.halaman_unggah.file_path
            data = pd.read_csv(file_path)

            # Validasi kolom
            if 'Close' not in data.columns:
                tk.messagebox.showerror("Error", "File harus memiliki kolom 'Close'.")
                return

            # Pisahkan train-test (80:20) tanpa shuffle
            train_data, test_data = train_test_split(data['Close'], test_size=0.2, shuffle=False)
            
            # Simpan data test dalam DataFrame
            self.test_data2 = pd.DataFrame(test_data, columns=['Close']).reset_index(drop=True)

            # Tampilkan hasil di notifikasi
            self.notifikasi.config(
                text="Data uji berhasil diambil! Data memiliki "
                    f"{len(self.test_data2)} baris."
            )
            self.notifikasi.config(fg="green")

            print("Test Data:")
            print(self.test_data2)

        except Exception as e:
            tk.messagebox.showerror("Error", f"Terjadi kesalahan saat memproses data uji:\n{e}")

    def hitung_var(self):
        if not hasattr(self, 'test_data2') or self.test_data2.empty:
            tk.messagebox.showerror("Error", "Data uji belum tersedia. Klik 'Ambil Data Uji' terlebih dahulu.")
            return

        try:
            # Gunakan data uji untuk menghitung returns
            test_close_prices = self.test_data2['Close'].values
            returns = np.diff(test_close_prices) / test_close_prices[:-1]

            # Hitung VaR (5% quantile)
            var_5 = np.percentile(returns, 5)

            # Tampilkan hasil pada widget hasil_var
            self.hasil_var.config(state="normal")
            self.hasil_var.delete("1.0", "end")
            self.hasil_var.insert("end", f"Value at Risk (5%): {var_5:.4%}\n")
            self.hasil_var.config(state="disabled")

            # Update histogram pada grafik
            self.ax.clear()
            self.ax.hist(returns, bins=30, color='blue', alpha=0.7)
            self.ax.axvline(var_5, color='red', linestyle='--', label=f'VaR (5%): {var_5:.4%}')
            self.ax.legend()
            self.ax.set_title("Histogram Returns")
            self.ax.set_xlabel("Returns")
            self.ax.set_ylabel("Frequency")

            # Refresh Canvas
            self.canvas.draw()

        except Exception as e:
            tk.messagebox.showerror("Error", f"Terjadi kesalahan: {e}")

# Menjalankan aplikasi
if __name__ == "__main__":
    app = AplikasiSaham()
    app.mainloop()
