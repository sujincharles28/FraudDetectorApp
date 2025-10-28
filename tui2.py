# tui2.py (Combined TUI - Integrated Logic - Replaced Spacer)

import sys
import os
from functools import partial
import re
import datetime

# --- Imports for integrated logic ---
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
# ------------------------------------

# --- Imports for OCR ---
try:
    import pytesseract
    from PIL import Image
except ImportError:
    print("[STARTUP WARNING] OCR libraries (pytesseract, Pillow) not found.", file=sys.stderr)
    print("Image detection will fail. Install them: pip install pytesseract Pillow", file=sys.stderr)
    pytesseract = None
    Image = None
# -----------------------

from textual.app import App, ComposeResult
from textual.containers import Vertical, Horizontal, Container
# --- MODIFIED: Removed Spacer ---
from textual.widgets import Header, Footer, Button, Static, TextArea, LoadingIndicator, DirectoryTree, Label
from textual.reactive import var
from textual import events
from textual.message import Message

# --- Setup Logging ---
LOG_FILE = "tui2_run.log"
with open(LOG_FILE, "w") as f:
    f.write(f"--- Combined TUI Log started at {datetime.datetime.now()} ---\n")

def log_message(message, exc_info=False):
    """Helper function to write messages to the log file."""
    try:
        with open(LOG_FILE, "a") as f:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            f.write(f"{timestamp} - {message}\n")
            if exc_info:
                import traceback
                traceback.print_exc(file=f)
    except Exception as e:
        print(f"!!! FAILED TO WRITE LOG: {e} !!!", file=sys.stderr)
# ---------------------

# ====================================================================
# Global Settings
# ====================================================================
MODEL_PATH = "saved_model_balanced"
BEST_THRESHOLD_PATH = os.path.join(MODEL_PATH, "best_threshold.txt")
DEFAULT_THRESHOLD = 0.53

# State constants for managing views
STATE_WELCOME = 0
STATE_MODE_SELECTION = 1
STATE_TEXT_INPUT = 2
STATE_IMAGE_INPUT = 3
STATE_ERROR = 99

# ====================================================================
# Integrated OCR Function
# ====================================================================
def extract_text_from_image(image_path):
    """Uses pytesseract to extract text from an image file."""
    if not pytesseract or not Image:
        log_message("TUI2 OCR ERROR: pytesseract or Pillow not installed.")
        raise ImportError("Required OCR libraries (pytesseract, Pillow) are missing.")
    log_message(f"TUI2 OCR: Attempting to process image: {image_path}")
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        log_message(f"TUI2 OCR: OCR successful for {os.path.basename(image_path)}.")
        return text
    except pytesseract.TesseractNotFoundError:
        log_message("TUI2 OCR ERROR: Tesseract executable not found in PATH.")
        raise pytesseract.TesseractNotFoundError("Tesseract executable not found. Ensure it's installed and in PATH.")
    except FileNotFoundError:
        log_message(f"TUI2 OCR ERROR: Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        log_message(f"TUI2 OCR ERROR: Failed to process image with OCR: {e}", exc_info=True)
        raise RuntimeError(f"Failed to process image with OCR: {e}")

# ====================================================================
# Main Application Class
# ====================================================================
class FraudApp(App):
    """Combined TUI for Text & Image Fraud Detection (Integrated Logic)."""

    CSS_PATH = "tui2.css"

    # --- Store Model Assets Here ---
    tokenizer = None
    model = None
    threshold = DEFAULT_THRESHOLD
    model_load_error = None
    # -------------------------------

    current_state = var(STATE_WELCOME)
    selected_image_path = var("", init=False)

    # --- Custom Message for Worker Completion ---
    class PredictionComplete(Message):
        def __init__(self, result_text: str, target_state: int) -> None:
            self.result_text = result_text
            self.target_state = target_state
            super().__init__()
    # --------------------------------------------

    def compose(self) -> ComposeResult:
        log_message("TUI2: compose() called")
        yield Header(name="FraudFinder - Fake Job Detection")

        with Container(id="app-body"):
            # --- Error View ---
            with Vertical(id="error-container", classes="view-hidden"):
                 yield Label("[b red]CRITICAL ERROR[/b red]", classes="view-title")
                 yield Static(id="error_message", classes="instructions")
                 yield Button("Quit", id="quit_error", variant="error")

            # --- Welcome View ---
            with Vertical(id="welcome-container", classes="view-hidden"):
                # --- MODIFIED: Replaced Spacer with Static() ---
                with Horizontal(classes="nav-bar"):
                    yield Static() # This acts as a spacer
                    yield Button("Quit ->", id="quit_button", classes="quit-button")
                # ---------------------------------------------
                yield Label("[b]Welcome to Fake Job Detection System![/b]", classes="view-title")
                yield Static("""
                [#e0e0e0]This application helps you detect potentially fraudulent job postings.
                Please read the instructions below to get started:
                [/]

                [b]Instructions:[/b]
                1.  [#e0e0e0]Use the 'Start' button to proceed.[/]
                2.  [#e0e0e0]Choose 'Text Input' or 'Image Input'.[/]
                3.  [#e0e0e0]Text: Paste description, click 'Detect'.[/]
                4.  [#e0e0e0]Image: Select image file, click 'Extract & Detect'.[/]
                5.  [#e0e0e0]Use 'Back' buttons or 'Quit' button to exit.[/]
                """, classes="instructions")
                yield Button("Start", id="start_button", variant="primary", classes="start-button")

            # --- Mode Selection View ---
            with Vertical(id="mode-selection-container", classes="view-hidden"):
                # --- MODIFIED: Replaced Spacer with Static() ---
                with Horizontal(classes="nav-bar"):
                    yield Button("<- Back", id="back_to_welcome", classes="back-button")
                    yield Static() # This acts as a spacer
                    yield Button("Quit ->", id="quit_button", classes="quit-button")
                # -------------------------------------
                yield Label("[b]Select Input Method:[/b]", classes="view-title mode-prompt")
                with Horizontal(id="mode-buttons"):
                    yield Button("Text Input", id="text_mode", variant="default", classes="mode-button")
                    yield Button("Image Input", id="image_mode", variant="default", classes="mode-button")

            # --- Text Input View ---
            with Vertical(id="text-input-container", classes="view-hidden"):
                # --- MODIFIED: Replaced Spacer with Static() ---
                with Horizontal(classes="nav-bar"):
                    yield Button("<- Back", id="back_to_mode_from_text", classes="back-button")
                    yield Static() # This acts as a spacer
                    yield Button("Quit ->", id="quit_button", classes="quit-button")
                # -------------------------------------
                yield Static("[b]Paste Job Description:[/b]", classes="input-prompt")
                yield TextArea(
                    id="job_input", language="text", theme="monokai",
                    placeholder="Paste the full job description here..."
                )
                with Horizontal(id="action-buttons-text"):
                    yield Button("Detect Fraud", id="detect_text_button", variant="primary", classes="action-button")
                    yield Button("Clear", id="clear_text_button", variant="warning", classes="action-button")
                yield LoadingIndicator(id="loading_text", classes="loading-indicator")
                yield Static(id="results_text_output", classes="results-output")

            # --- Image Input View ---
            with Vertical(id="image-input-container", classes="view-hidden"):
                # --- MODIFIED: Replaced Spacer with Static() ---
                with Horizontal(classes="nav-bar"):
                    yield Button("<- Back", id="back_to_mode_from_image", classes="back-button")
                    yield Static() # This acts as a spacer
                    yield Button("Quit ->", id="quit_button", classes="quit-button")
                # -------------------------------------
                yield Static("[b]Select Image File:[/b]", classes="input-prompt")
                yield DirectoryTree(os.path.expanduser("~"))
                yield Static(id="selected_file_display", classes="selected-file-display")
                with Horizontal(id="action-buttons-image"):
                    yield Button("Extract & Detect", id="detect_image_button", variant="primary", classes="action-button")
                    yield Button("Clear Selection", id="clear_image_button", variant="warning", classes="action-button")
                yield LoadingIndicator(id="loading_image", classes="loading-indicator")
                yield Static(id="ocr_output", classes="ocr-output view-hidden")
                yield Static(id="results_image_output", classes="results-output")

        yield Footer()

    # --- Model Loading Logic ---
    def load_model_assets(self):
        """Loads the tokenizer, model, and threshold. Stores errors."""
        log_message(f"TUI2: Attempting to load model assets from {MODEL_PATH}")
        try:
            if 'torch' not in sys.modules or 'transformers' not in sys.modules:
                 raise ImportError("Core ML libraries (torch, transformers) not found.")
            self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
            self.model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
            self.model.eval()

            if os.path.exists(BEST_THRESHOLD_PATH):
                try:
                    with open(BEST_THRESHOLD_PATH, "r") as f: self.threshold = float(f.read().strip())
                    log_message(f"TUI2: Loaded threshold {self.threshold:.3f}")
                except ValueError:
                    self.threshold = DEFAULT_THRESHOLD
                    log_message(f"[WARNING] TUI2: Could not parse threshold. Using default: {self.threshold:.3f}")
            else:
                 self.threshold = DEFAULT_THRESHOLD
                 log_message(f"[WARNING] TUI2: Threshold file not found. Using default: {self.threshold:.3f}")
            log_message("TUI2: Model assets loaded successfully")
            self.model_load_error = None; return True
        except ImportError as e:
             self.model_load_error = f"Missing core ML libraries: {e}."; log_message(f"[ERROR] TUI2: {self.model_load_error}"); return False
        except Exception as e:
            self.model_load_error = f"Failed to load model/tokenizer from '{MODEL_PATH}':\n{e}"; log_message(f"[ERROR] TUI2: {self.model_load_error}", exc_info=True); return False

    # --- Integrated Prediction Logic (Original detect_fraud.py) ---
    def _perform_prediction(self, text_to_analyze: str) -> tuple[float, str]:
        """Performs fraud detection using loaded model (NO CLEANING)."""
        log_message("TUI2 PRED: Starting prediction logic")
        if not self.model or not self.tokenizer:
             log_message("TUI2 PRED ERROR: Model or Tokenizer not available.")
             return -1.0, "ERROR"
        if not text_to_analyze or not text_to_analyze.strip():
             log_message("TUI2 PRED ERROR: No text provided for analysis.")
             return -1.0, "ERROR"
        try:
            log_message("TUI2 PRED: Tokenizing input text...")
            inputs = self.tokenizer(text_to_analyze, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            log_message("TUI2 PRED: Text tokenized.")
        except Exception as e:
             log_message(f"TUI2 PRED ERROR: Tokenization failed: {e}", exc_info=True)
             return -1.0, "ERROR"
        try:
            log_message("TUI2 PRED: Running model inference...")
            with torch.no_grad(): outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prob = probabilities[0][1].item()
            log_message(f"TUI2 PRED: Inference complete. Raw probability={prob}")
            prediction_string = "FRAUDULENT" if prob >= self.threshold else "LEGITIMATE"
            log_message(f"TUI2 PRED: Final prediction='{prediction_string}' using threshold={self.threshold}")
            return prob, prediction_string
        except Exception as e:
            log_message(f"TUI2 PRED ERROR: Model inference failed: {e}", exc_info=True)
            return -1.0, "ERROR"

    # --- Worker Functions ---
    def _run_text_prediction_worker(self, text: str):
        """Worker function for text input."""
        log_message("TUI2 WORKER (Text): Starting prediction")
        display_result = "[bold red]ERROR: Text prediction failed.[/bold red]"
        try:
             prob, prediction = self._perform_prediction(text)
             log_message(f"TUI2 WORKER (Text): Prediction complete (Prob: {prob:.3f}, Result: {prediction})")
             # Format result - REMOVED PROBABILITY
             if prediction == "ERROR":
                 display_result = "[bold red]PREDICTION ERROR:[/bold red] Check log."
             elif prediction == "FRAUDULENT":
                 display_result = (f"[bold red on #2C0000] PREDICTION: {prediction} [/]")
             else: # LEGITIMATE
                 display_result = (f"[bold green on #0A2B00] PREDICTION: {prediction} [/]")
             self.call_from_thread(self.update_results, display_result, STATE_TEXT_INPUT)
        except Exception as e:
             error_text = f"[bold red]WORKER ERROR:[/bold red]\n{str(e)}"
             log_message(f"TUI2 WORKER (Text): Unexpected Error: {e}", exc_info=True)
             self.call_from_thread(self.update_results, error_text, STATE_TEXT_INPUT)

    def _run_image_ocr_and_prediction_worker(self, image_path: str):
        """Worker function for image OCR and integrated prediction."""
        log_message(f"TUI2 WORKER (Image): Starting OCR for image: {image_path}")
        display_result = "[bold red]ERROR: Image processing failed.[/bold red]"
        ocr_text_preview = ""
        try:
            ocr_text = extract_text_from_image(image_path)
            log_message("TUI2 WORKER (Image): OCR Complete.")
            if not ocr_text.strip():
                 log_message("TUI2 WORKER: OCR Failed - No text extracted.")
                 display_result = "[bold red]OCR Failed:[/bold red] No text extracted."
                 self.call_from_thread(self.update_results, display_result, STATE_IMAGE_INPUT)
                 return

            ocr_text_preview = f"[#957DAD]Extracted Text Preview:[/]\n{ocr_text[:200].strip()}..."
            self.call_from_thread(self._update_and_show_ocr_preview, ocr_text_preview)

            log_message("TUI2 WORKER (Image): Starting prediction on OCR text")
            prob, prediction = self._perform_prediction(ocr_text)
            log_message(f"TUI2 WORKER (Image): Prediction complete (Prob: {prob:.3f}, Result: {prediction})")

            # Format result - REMOVED PROBABILITY
            if prediction == "ERROR": display_result = "[bold red]PREDICTION ERROR:[/bold red] Check log."
            elif prediction == "FRAUDULENT": display_result = (f"[bold red on #2C0000] PREDICTION: {prediction} [/]")
            else: display_result = (f"[bold green on #0A2B00] PREDICTION: {prediction} [/]")
            self.call_from_thread(self.update_results, display_result, STATE_IMAGE_INPUT)

        except ImportError as e: error_text = f"[bold red]Missing Library:[/bold red] {e}"; log_message(f"TUI2 WORKER ERROR: {error_text}"); self.call_from_thread(self.update_results, error_text, STATE_IMAGE_INPUT)
        except (pytesseract.TesseractNotFoundError if pytesseract else Exception): error_text = "[bold red]Tesseract Not Found.[/bold red]"; log_message(f"TUI2 WORKER ERROR: {error_text}"); self.call_from_thread(self.update_results, error_text, STATE_IMAGE_INPUT)
        except FileNotFoundError as e: error_text = f"[bold red]File Not Found:[/bold red] {e}"; log_message(f"TUI2 WORKER ERROR: {error_text}"); self.call_from_thread(self.update_results, error_text, STATE_IMAGE_INPUT)
        except RuntimeError as e: error_text = f"[bold red]Image Processing Error:[/bold red] {e}"; log_message(f"TUI2 WORKER ERROR: {error_text}", exc_info=True); self.call_from_thread(self.update_results, error_text, STATE_IMAGE_INPUT)
        except Exception as e: error_text = f"[bold red]Unexpected Worker Error:[/bold red] {str(e)}"; log_message(f"TUI2 WORKER ERROR: Unexpected Exception: {e}", exc_info=True); self.call_from_thread(self.update_results, error_text, STATE_IMAGE_INPUT)

    # --- Helper to update OCR preview from worker ---
    def _update_and_show_ocr_preview(self, preview_text: str):
        """Safely updates the OCR preview box and makes it visible."""
        try:
            ocr_widget = self.query_one("#ocr_output")
            ocr_widget.update(preview_text)
            ocr_widget.set_class(False, "view-hidden")
            ocr_widget.display = True
            ocr_widget.refresh() # Add refresh
            log_message("TUI2 MAIN: OCR preview updated and shown.")
        except Exception as e:
            log_message(f"TUI2 MAIN ERROR: updating OCR preview: {e}", exc_info=True)
    # ------------------------------------------------

    # --- UI State Management ---
    def watch_current_state(self, old_state: int, new_state: int) -> None:
        """Shows/hides views based on current_state."""
        log_message(f"TUI2: watch_current_state: {old_state} -> {new_state}")
        self.set_view_display(STATE_ERROR, new_state == STATE_ERROR)
        self.set_view_display(STATE_WELCOME, new_state == STATE_WELCOME)
        self.set_view_display(STATE_MODE_SELECTION, new_state == STATE_MODE_SELECTION)
        self.set_view_display(STATE_TEXT_INPUT, new_state == STATE_TEXT_INPUT)
        self.set_view_display(STATE_IMAGE_INPUT, new_state == STATE_IMAGE_INPUT)

        try:
            if new_state == STATE_TEXT_INPUT:
                self.query_one("#loading_text").display = False; self.query_one("#detect_text_button").disabled = False
                self.query_one("#results_text_output").update(""); self.set_timer(0.1, self.query_one(TextArea).focus)
            elif new_state == STATE_IMAGE_INPUT:
                self.query_one("#loading_image").display = False; self.query_one("#detect_image_button").disabled = not self.selected_image_path
                self.query_one("#selected-file-display").update(f"[#957DAD]Selected Image:[/][b] {os.path.basename(self.selected_image_path)}[/b]" if self.selected_image_path else "[#957DAD]No image selected yet.[/]")
                self.query_one("#ocr_output").update(""); self.query_one("#ocr_output").display = False
                self.query_one("#results_image_output").update(""); self.set_timer(0.1, self.query_one(DirectoryTree).focus)
            elif new_state == STATE_ERROR:
                 error_msg = getattr(self, 'model_load_error', "Unknown error."); self.query_one("#error_message").update(error_msg or "Unknown error.")
        except Exception as e: log_message(f"TUI2 ERROR: during UI reset in watch_current_state: {e}", exc_info=True)


    def set_view_display(self, state: int, display: bool) -> None:
        """Helper to show/hide the main view containers."""
        container_ids = { STATE_ERROR: "#error-container", STATE_WELCOME: "#welcome-container", STATE_MODE_SELECTION: "#mode-selection-container", STATE_TEXT_INPUT: "#text-input-container", STATE_IMAGE_INPUT: "#image-input-container" }
        container_id = container_ids.get(state)
        if container_id:
            try:
                widget = self.query_one(container_id); widget.set_class(not display, "view-hidden"); widget.display = display
            except Exception as e: log_message(f"TUI2 ERROR: setting display for {container_id}: {e}")

    # --- Event Handlers ---
    def on_mount(self) -> None:
        """Load model assets and configure initial UI state."""
        log_message("TUI2: on_mount() called")
        if not self.load_model_assets(): self.current_state = STATE_ERROR
        else: self.current_state = STATE_WELCOME

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id; log_message(f"TUI2: Button pressed: {button_id}")
        
        if button_id == "quit_error" or button_id == "quit_button": 
            log_message("TUI2: Quit requested."); self.exit()
        elif button_id == "start_button": self.current_state = STATE_ERROR if self.model_load_error else STATE_MODE_SELECTION
        elif button_id == "text_mode": self.current_state = STATE_TEXT_INPUT
        elif button_id == "image_mode": self.current_state = STATE_IMAGE_INPUT
        elif button_id in ("back_to_welcome", "back_to_mode_from_text", "back_to_mode_from_image"):
             if self.current_state in (STATE_TEXT_INPUT, STATE_IMAGE_INPUT): self.current_state = STATE_MODE_SELECTION
             elif self.current_state == STATE_MODE_SELECTION: self.current_state = STATE_WELCOME
        
        elif button_id == "detect_text_button":
            job_text = self.query_one(TextArea).text
            if not job_text.strip(): self.query_one("#results_text_output").update("[bold red]Paste text first.[/]"); return
            self.query_one("#loading_text").display = True; self.query_one("#detect_text_button").disabled = True
            self.query_one("#results_text_output").update("[italic blue]Analyzing...[/]")
            log_message("TUI2: Calling worker for text prediction...")
            self.run_worker(partial(self._run_text_prediction_worker, job_text), exclusive=True, thread=True)
        elif button_id == "clear_text_button":
             self.query_one(TextArea).clear(); self.query_one("#results_text_output").update(""); self.query_one(TextArea).focus()
        elif button_id == "detect_image_button":
            if self.model_load_error: self.query_one("#results_image_output").update(f"[bold red]Model Error.[/]"); return
            if not pytesseract or not Image: self.query_one("#results_image_output").update(f"[bold red]OCR Libs Missing.[/]"); return
            if not self.selected_image_path or not os.path.exists(self.selected_image_path): self.query_one("#results_image_output").update("[bold red]Select valid image.[/]"); return
            self.query_one("#loading_image").display = True; self.query_one("#detect_image_button").disabled = True
            self.query_one("#ocr_output").update(""); self.query_one("#ocr_output").display = False
            self.query_one("#results_image_output").update("[italic blue]Processing image...[/]")
            log_message(f"TUI2: Calling worker for image OCR & prediction ({self.selected_image_path})...")
            self.run_worker(partial(self._run_image_ocr_and_prediction_worker, self.selected_image_path), exclusive=True, thread=True)
        elif button_id == "clear_image_button":
             self.selected_image_path = "";
             try:
                 self.query_one("#selected-file-display").update("[#957DAD]No image selected yet.[/]")
                 self.query_one("#ocr_output").update(""); self.query_one("#ocr_output").display = False
                 self.query_one("#results_image_output").update("")
                 self.query_one("#detect_image_button").disabled = True
             except Exception as e: log_message(f"TUI2 ERROR: Clearing image UI elements failed: {e}")


    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        """Handle file selection, validate, and update state."""
        file_path = str(event.path); log_message(f"TUI2: File selected in tree: {file_path}")
        valid_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')
        error_widget = self.query_one("#results_image_output"); detect_button = self.query_one("#detect_image_button")
        
        if not os.path.isfile(file_path): 
            error_widget.update("[bold red]Error: Not a file.[/]"); self.selected_image_path = ""; 
            log_message("TUI2: Selection not a file, disabling button."); detect_button.disabled = True; return
        if not file_path.lower().endswith(valid_extensions): 
            error_widget.update("[bold red]Error: Not valid image type.[/]"); self.selected_image_path = ""; 
            log_message("TUI2: Selection not valid image, disabling button."); detect_button.disabled = True; return
        
        log_message("TUI2: Valid image selected. Enabling button.")
        self.selected_image_path = file_path; error_widget.update(""); detect_button.disabled = False
        log_message(f"TUI2: Button '#detect_image_button' disabled state: {detect_button.disabled}")
        self.query_one("#ocr_output").update(""); self.query_one("#ocr_output").display = False
        self.query_one("#results_image_output").update("")


    # --- UI Update Method (Called via call_from_thread) ---
    def update_results(self, result_text: str, target_state: int) -> None:
        """Update the correct results box safely."""
        log_message(f"TUI2 MAIN: update_results called for state {target_state}")
        results_widget_id = "#results_text_output" if target_state == STATE_TEXT_INPUT else "#results_image_output"
        loading_widget_id = "#loading_text" if target_state == STATE_TEXT_INPUT else "#loading_image"
        detect_button_id = "#detect_text_button" if target_state == STATE_TEXT_INPUT else "#detect_image_button"

        try:
            loading_widget = self.query_one(loading_widget_id); loading_widget.display = False
            detect_button = self.query_one(detect_button_id); detect_button.disabled = False
            if target_state == STATE_IMAGE_INPUT and not self.selected_image_path: detect_button.disabled = True 

            results_widget = self.query_one(results_widget_id); results_widget.update(result_text); results_widget.refresh()
            log_message("TUI2 MAIN: UI updated (and refreshed).")
        except Exception as e:
             log_message(f"TUI2 MAIN ERROR: Failed during update_results for state {target_state}: {e}", exc_info=True)


if __name__ == "__main__":
    log_message("--- TUI2: Starting Combined App ---")
    app = FraudApp()
    try: app.run()
    except Exception as e: log_message(f"--- TUI2: App crashed: {e} ---", exc_info=True); raise
    finally: log_message("--- TUI2: App finished ---")

