import tkinter as tk
from tkinter import messagebox, simpledialog
import subprocess
import os
import pickle
import cv2
import shutil
import tensorflow as tf

# === Paths ===
project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
scripts_folder = os.path.join(project_folder, 'scripts')
dataset_folder = os.path.join(project_folder, 'dataset')

# === App ===
class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 Face Recognition GUI")
        self.root.geometry("420x430")
        self.root.configure(bg="#f0f4f8")
        self.model_type = tk.StringVar(value="face_recognition")

        self.user_listbox = None
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.root, text="🎯 Face Recognition Interface", font=("Arial", 16, "bold"), bg="#f0f4f8").pack(pady=10)

        # Model selection
        frame_model = tk.Frame(self.root, bg="#f0f4f8")
        frame_model.pack()
        tk.Label(frame_model, text="Choose Model:", bg="#f0f4f8", font=("Arial", 12)).pack(side="left", padx=5)
        tk.OptionMenu(frame_model, self.model_type, "face_recognition", "custom_model").pack(side="left")

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#f0f4f8")
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="👤 Register New User", width=25, command=self.preview_and_register).pack(pady=5)
        tk.Button(btn_frame, text="🔄 Update Model", width=25, command=self.update_model).pack(pady=5)
        tk.Button(btn_frame, text="📷 Start Live Recognition", width=25, command=self.start_recognition).pack(pady=5)
        tk.Button(btn_frame, text="🗑️ Delete Selected User", width=25, command=self.delete_selected_user).pack(pady=5)

        # === User list display ===
        tk.Label(self.root, text="📋 Registered Users:", font=("Arial", 11, "bold"), bg="#f0f4f8").pack(pady=(10, 0))
        self.user_listbox = tk.Listbox(self.root, height=6, width=30)
        self.user_listbox.pack()
        self.user_listbox.bind("<Double-Button-1>", self.confirm_delete_on_click)

        # === Refresh when model changes ===
        self.model_type.trace("w", lambda *args: self.refresh_user_list())

        # Initial load
        self.refresh_user_list()

    def run_script(self, script_name, extra_env=None):
        script_path = os.path.join(scripts_folder, script_name)
        env = os.environ.copy()
        if extra_env:
            env.update(extra_env)
        subprocess.run(["python", script_path], env=env)

    def preview_and_register(self):
        # 1. Camera Preview
        cap = cv2.VideoCapture(0)
        cv2.namedWindow("📷 Preview - Press SPACE to continue", cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow("📷 Preview - Press SPACE to continue", frame)
            key = cv2.waitKey(1)
            if key == 32:  # SPACE to confirm
                break
            elif key == 27:  # ESC to cancel
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()
        cv2.destroyAllWindows()

        # 2. Ask for Name
        name = simpledialog.askstring("Enter Name", "Enter new user's name (no spaces):")
        if not name:
            return

        # 3. Trigger Register Script
        self.run_script("register_user.py", extra_env={"NEW_USER_NAME": name})
        self.refresh_user_list()

    def update_model(self):
        script = "update_model.py" if self.model_type.get() == "face_recognition" else "update_model2.py"
        self.run_script(script)
        self.refresh_user_list()

    def start_recognition(self):
        script = "recognize_live.py" if self.model_type.get() == "face_recognition" else "recognize_live2.py"
        self.run_script(script)

    def get_pickle_path(self):
        if self.model_type.get() == "face_recognition":
            return os.path.join(project_folder, "encodings", "encodings.pickle")
        else:
            return os.path.join(project_folder, "embeddings", "custom_embeddings.pickle")

    def refresh_user_list(self):
        self.user_listbox.delete(0, tk.END)
        pickle_path = self.get_pickle_path()

        if os.path.exists(pickle_path):
            try:
                with open(pickle_path, "rb") as f:
                    data = pickle.load(f)
                if "names" in data:
                    for name in sorted(set(data["names"])):
                        self.user_listbox.insert(tk.END, name)
                else:
                    self.user_listbox.insert(tk.END, "(No users found)")
            except Exception as e:
                self.user_listbox.insert(tk.END, f"Error: {e}")
        else:
            self.user_listbox.insert(tk.END, "(No users found)")

    def delete_selected_user(self):
        selection = self.user_listbox.curselection()
        if not selection:
            messagebox.showinfo("Select User", "Please select a user to delete.")
            return

        username = self.user_listbox.get(selection[0])
        confirm = messagebox.askyesno("Confirm", f"Delete user '{username}' from system?")
        if confirm:
            self.delete_user(username)

    def confirm_delete_on_click(self, event):
        self.delete_selected_user()

    def delete_user(self, username):
        pickle_path = self.get_pickle_path()
        found = False

        # === 1. Delete dataset folder ===
        user_folder = os.path.join(dataset_folder, username)
        if os.path.exists(user_folder):
            shutil.rmtree(user_folder)
            found = True

        # === 2. Remove from pickle ===
        if os.path.exists(pickle_path):
            with open(pickle_path, "rb") as f:
                data = pickle.load(f)

            if username in data.get("names", []):
                if "embeddings" in data:
                    zipped = zip(data["names"], data["embeddings"])
                    key = "embeddings"
                elif "encodings" in data:
                    zipped = zip(data["names"], data["encodings"])
                    key = "encodings"
                else:
                    messagebox.showerror("Error", "⚠️ Pickle file structure not recognized.")
                    return

                new_names = []
                new_vectors = []
                for name, vec in zipped:
                    if name != username:
                        new_names.append(name)
                        new_vectors.append(vec)

                with open(pickle_path, "wb") as f:
                    pickle.dump({"names": new_names, key: new_vectors}, f)

                found = True

        # === 3. Update user_list.txt if using face_recognition ===
        if self.model_type.get() == "face_recognition":
            user_list_path = os.path.join(project_folder, "encodings", "user_list.txt")
            if os.path.exists(user_list_path):
                with open(user_list_path, "r") as f:
                    users = [line.strip() for line in f if line.strip() != username]
                with open(user_list_path, "w") as f:
                    for name in users:
                        f.write(f"{name}\n")

        # === 4. Final status ===
        if found:
            messagebox.showinfo("Success", f"✅ User '{username}' deleted.")
        else:
            messagebox.showwarning("Not Found", f"⚠️ User '{username}' not found.")

        self.refresh_user_list()

# === Launch App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()
