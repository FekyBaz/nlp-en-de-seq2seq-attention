"""
Compiles the comprehensive academic report into a publication-quality PDF
using ReportLab with university branding, structured headings, mathematical
specifications, tables, embedded figures, and qualitative error analysis.
"""

import os
import json
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether,
    HRFlowable,
    PageBreak,
)
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas to dynamically compute and print total page count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(
                54,
                750,
                "Alexandria National University | NLP Fall 2026 — Assignment 3: NMT with LSTM & Attention",
            )
            self.setStrokeColor(colors.HexColor("#dddddd"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "Mohamed Baz | github.com/FekyBaz/nlp-en-de-seq2seq-attention")
        self.setStrokeColor(colors.HexColor("#dddddd"))
        self.setLineWidth(0.5)
        self.line(54, 48, 558, 48)

        self.restoreState()


def build_pdf_report(
    output_pdf_path="report/report.pdf",
    metrics_path="outputs/results/metrics.json",
    length_path="outputs/results/length_analysis.json",
    qual_path="outputs/results/qualitative_table.csv",
):
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    primary_color = colors.HexColor("#1a365d")
    secondary_color = colors.HexColor("#2b6cb0")
    text_dark = colors.HexColor("#2d3748")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=0,
        spaceAfter=6,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=secondary_color,
        spaceAfter=14,
    )

    meta_style = ParagraphStyle(
        "DocMeta",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#4a5568"),
        spaceAfter=14,
    )

    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )

    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )

    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=text_dark,
        spaceAfter=6,
    )

    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3,
    )

    code_style = ParagraphStyle(
        "Code_Custom",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#2c3e50"),
        backColor=colors.HexColor("#f7fafc"),
        borderPadding=4,
        spaceAfter=6,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Neural Machine Translation with LSTM and Attention", title_style))
    story.append(
        Paragraph(
            "Comparative Empirical Analysis: Fixed-Length Context Vector Bottleneck vs. Dynamic Bahdanau Alignment",
            subtitle_style,
        )
    )
    story.append(
        Paragraph(
            "<b>Course:</b> Natural Language Processing (Fall 2026) | <b>Alexandria National University</b><br/>"
            "<b>Author:</b> Mohamed Baz | <b>Repository:</b> github.com/FekyBaz/nlp-en-de-seq2seq-attention",
            meta_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=primary_color, spaceAfter=10))

    # Section 1: Executive Summary & Objective
    story.append(Paragraph("1. Executive Summary & Problem Formulation", h1_style))
    story.append(
        Paragraph(
            "Neural Machine Translation (NMT) maps variable-length source sequences to target sequences. "
            "In canonical Sequence-to-Sequence (Seq2Seq) LSTM architectures (Sutskever et al., 2014), the encoder "
            "compresses the entire source sentence into a single fixed-length hidden vector (<i>h<sub>T</sub></i>, <i>c<sub>T</sub></i>). "
            "This creates an <b>information bottleneck</b>: as input sentence length increases, the model's memory degrades, "
            "causing missed words, hallucinated repetitions, and syntactic breakdowns.",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "To overcome this bottleneck, Bahdanau et al. (2014) proposed <b>Additive Attention</b>. A Bidirectional LSTM outputs "
            "annotations for every source token, and an attention mechanism computes dynamic alignment weights (<i>&alpha;<sub>t,s</sub></i>) "
            "and time-varying context vectors (<i>c<sub>t</sub></i>) at every decoding step. This project implements, trains, and empirically "
            "contrasts both architectures on English-to-German translation.",
            body_style,
        )
    )

    # Section 2: Preprocessing Pipeline
    story.append(Paragraph("2. Dataset & Preprocessing Pipeline", h1_style))
    story.append(
        Paragraph(
            "Experiments are conducted on the benchmark <b>Multi30k</b> English-German parallel corpus (10,000 train, 1,014 validation, 1,000 test). "
            "Key engineering safeguards implemented in the preprocessing module include:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "&bull; <b>German Character Preservation:</b> Strict Unicode NFC normalization preserving umlauts (&auml;, &ouml;, &uuml;) "
            "and Eszett (&szlig;) to avoid semantic corruption (e.g., <i>schon</i> [already] vs. <i>sch&ouml;n</i> [beautiful]).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "&bull; <b>Punctuation Isolation & Special Tokens:</b> Whitespace-isolated punctuation and standardized reserved indices: "
            "<code>&lt;pad&gt;=0</code>, <code>&lt;sos&gt;=1</code>, <code>&lt;eos&gt;=2</code>, <code>&lt;unk&gt;=3</code> (min_freq=2).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "&bull; <b>Dynamic Batch Collation:</b> Sequence padding dynamically tailored to batch-level maximum length rather than global maximum.",
            bullet_style,
        )
    )

    # Section 3: Model Architectures
    story.append(Paragraph("3. Model Architectures & Theoretical Foundations", h1_style))
    story.append(
        Paragraph(
            "To ensure rigorous, fair comparison, representational capacity is held strictly symmetric: "
            "Embedding dimension <i>d<sub>emb</sub> = 128</i>, Baseline LSTM hidden dimension <i>d<sub>hid</sub> = 256</i>. "
            "The Attention model employs a Bidirectional LSTM with <i>d<sub>enc</sub> = 128</i> per direction (concatenated <i>H<sub>s</sub> &in; &reals;<sup>256</sup></i>) "
            "and Decoder hidden dimension <i>d<sub>dec</sub> = 256</i>.",
            body_style,
        )
    )

    arch_table_data = [
        ["Component", "Baseline Seq2Seq (LSTM)", "Seq2Seq with Bahdanau Attention"],
        ["Encoder", "1-Layer Unidirectional LSTM (dim=256)", "1-Layer Bidirectional LSTM (128 fwd + 128 bwd = 256)"],
        ["Context Representation", "Fixed final states: (h_S, c_S) in R^256", "Dynamic context vector: c_t = sum(alpha_{t,s} H_s) in R^256"],
        ["Attention Mechanism", "None (Static bottleneck)", "Additive: e_{t,s} = v_a^T tanh(W_a s_{t-1} + U_a H_s)"],
        ["Decoder", "Autoregressive LSTMCell / LSTM", "AttentionDecoder conditioned on [e_t; c_t]"],
        ["Loss & Optimization", "CrossEntropyLoss(ignore=PAD), Adam(1e-3)", "CrossEntropyLoss(ignore=PAD), Adam(1e-3)"],
    ]
    t_arch = Table(arch_table_data, colWidths=[110, 190, 204])
    t_arch.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ebf8ff")),
            ("TEXTCOLOR", (0, 0), (-1, 0), primary_color),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("LEADING", (0, 0), (-1, -1), 11),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e0")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(t_arch)
    story.append(Spacer(1, 10))

    # Section 4: Quantitative Results
    story.append(Paragraph("4. Quantitative Results & Comparative Analysis", h1_style))
    story.append(
        Paragraph(
            "Evaluation on the held-out test set demonstrates clear, across-the-board superiority for the Attention mechanism. "
            "All BLEU scores (BLEU-1 to BLEU-4) and ROUGE-L show marked improvements:",
            body_style,
        )
    )

    # Load metrics if available
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        b_m = metrics_data["Baseline"]
        a_m = metrics_data["Attention"]

        metrics_table_data = [
            ["Metric", "Baseline Seq2Seq", "Seq2Seq + Attention", "Absolute Gain (Delta)"],
            ["BLEU-1 (Unigram)", f"{b_m['BLEU-1']:.2f}", f"{a_m['BLEU-1']:.2f}", f"+{a_m['BLEU-1'] - b_m['BLEU-1']:.2f}"],
            ["BLEU-2 (Bigram)", f"{b_m['BLEU-2']:.2f}", f"{a_m['BLEU-2']:.2f}", f"+{a_m['BLEU-2'] - b_m['BLEU-2']:.2f}"],
            ["BLEU-3 (Trigram)", f"{b_m['BLEU-3']:.2f}", f"{a_m['BLEU-3']:.2f}", f"+{a_m['BLEU-3'] - b_m['BLEU-3']:.2f}"],
            ["BLEU-4 (4-gram)", f"{b_m['BLEU-4']:.2f}", f"{a_m['BLEU-4']:.2f}", f"+{a_m['BLEU-4'] - b_m['BLEU-4']:.2f}"],
            ["ROUGE-L (F1)", f"{b_m['ROUGE-L']:.2f}", f"{a_m['ROUGE-L']:.2f}", f"+{a_m['ROUGE-L'] - b_m['ROUGE-L']:.2f}"],
        ]
        t_met = Table(metrics_table_data, colWidths=[120, 120, 134, 130])
        t_met.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6fffa")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#234e52")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b2f5ea")),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        story.append(t_met)
        story.append(Spacer(1, 10))

    # Loss curve image
    if os.path.exists("outputs/figures/loss_curves.png"):
        story.append(Image("outputs/figures/loss_curves.png", width=6.8 * inch, height=2.4 * inch))
        story.append(
            Paragraph("<i>Figure 1: Training and Validation Loss convergence curves across 10 epochs.</i>", meta_style)
        )
        story.append(Spacer(1, 6))

    # Section 5: Sentence Length Bottleneck Analysis
    story.append(Paragraph("5. Sentence-Length Bottleneck Analysis", h1_style))
    story.append(
        Paragraph(
            "To validate the core hypothesis of the assignment, the test set was stratified into three length tiers: "
            "<b>Short (3–6 words)</b>, <b>Medium (7–11 words)</b>, and <b>Long (12–18+ words)</b>. "
            "As depicted in Figure 2, the Baseline model's translation fidelity collapses on longer inputs as early tokens are overwritten. "
            "In stark contrast, the Attention mechanism preserves high BLEU and ROUGE-L across all length strata.",
            body_style,
        )
    )

    if os.path.exists("outputs/figures/length_analysis.png"):
        story.append(Image("outputs/figures/length_analysis.png", width=6.8 * inch, height=2.4 * inch))
        story.append(
            Paragraph(
                "<i>Figure 2: BLEU-4 and ROUGE-L comparative performance across sentence length strata.</i>", meta_style
            )
        )
        story.append(Spacer(1, 6))

    # Section 6: Attention Heatmaps
    story.append(PageBreak())
    story.append(Paragraph("6. Attention Alignment Visualizations & Linguistic Analysis", h1_style))
    story.append(
        Paragraph(
            "During greedy inference, the full 2D alignment matrix &alpha;<sub>t,s</sub> was recorded. "
            "Figure 3 presents heatmaps for diverse English-to-German structures:",
            body_style,
        )
    )

    if os.path.exists("outputs/figures/attention_sample_1.png") and os.path.exists("outputs/figures/attention_sample_4.png"):
        img_table = [
            [
                Image("outputs/figures/attention_sample_1.png", width=3.3 * inch, height=2.6 * inch),
                Image("outputs/figures/attention_sample_4.png", width=3.3 * inch, height=2.6 * inch),
            ],
            [
                Paragraph("<b>Sample 1:</b> <i>'A young boy in a blue jacket is climbing...'</i><br/>Monotonic diagonal with sharp focus on verb <i>klettert</i>.", meta_style),
                Paragraph("<b>Sample 4:</b> <i>'Two women sitting... while drinking coffee'</i><br/>Off-diagonal alignment reflecting German verb-final subordinate clause.", meta_style),
            ]
        ]
        t_img = Table(img_table, colWidths=[250, 250])
        t_img.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        story.append(t_img)
        story.append(Spacer(1, 8))

    # Section 7: Qualitative Error Analysis
    story.append(Paragraph("7. Qualitative Error Analysis & Error Taxonomy", h1_style))
    story.append(
        Paragraph(
            "A systematic comparison between model translations reveals four prominent error categories:",
            body_style,
        )
    )
    story.append(
        Paragraph(
            "1. <b>Content Omission & Truncation (Baseline):</b> When input length exceeds recurrent capacity, Baseline drops adjectives and prepositional modifiers.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "2. <b>Degenerative Repetition Loops (Baseline):</b> Baseline decoder hidden state frequently falls into repetitive limit cycles (e.g., <i>'im wasser im wasser'</i>). Attention eliminates this by continually conditioning on distinct source contexts.",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "3. <b>German Verb Reordering & Separable Verbs:</b> Attention successfully captures German subordinate verb-final placement (<i>'w&auml;hrend sie kaffee trinken'</i>) and split verbs (<i>'bereitet ... zu'</i>).",
            bullet_style,
        )
    )
    story.append(
        Paragraph(
            "4. <b>Inflection & Case Agreement:</b> Both models occasionally face challenges with subtle German adjective declension endings (nominative vs. accusative/dative), though Attention accurately resolves direct objects.",
            bullet_style,
        )
    )
    story.append(Spacer(1, 6))

    # Load qualitative table
    if os.path.exists(qual_path):
        q_df = pd.read_csv(qual_path)
        q_rows = [["ID", "Source (EN)", "Ground Truth (DE)", "Baseline Prediction", "Attention Prediction"]]
        for _, row in q_df.head(5).iterrows():
            q_rows.append([
                Paragraph(f"<b>{row['Sample ID']}</b>", body_style),
                Paragraph(str(row["Source (English)"]), body_style),
                Paragraph(str(row["Ground Truth (German)"]), body_style),
                Paragraph(f"<font color='#c53030'>{str(row['Baseline Prediction'])}</font>", body_style),
                Paragraph(f"<font color='#276749'><b>{str(row['Attention Prediction'])}</b></font>", body_style),
            ])
        t_qual = Table(q_rows, colWidths=[40, 115, 115, 115, 119])
        t_qual.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), primary_color),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ])
        )
        story.append(t_qual)
        story.append(Spacer(1, 10))

    # Section 8: Conclusion
    story.append(Paragraph("8. Conclusion", h1_style))
    story.append(
        Paragraph(
            "This empirical study demonstrates that the fixed-length context vector of baseline Seq2Seq represents a severe architectural bottleneck in NMT. "
            "Integrating Bahdanau Additive Attention over a Bidirectional LSTM provides robust gradient pathways, achieves substantially superior BLEU-1 to 4 and ROUGE-L, "
            "sustains high translation fidelity across long sentence lengths, and delivers interpretable 2D attention alignments faithful to English-German syntactic reordering.",
            body_style,
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[*] Report successfully compiled to {output_pdf_path}")


if __name__ == "__main__":
    build_pdf_report()
