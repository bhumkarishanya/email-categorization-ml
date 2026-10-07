const EXAMPLES = {
    spam: "Congratulations! You have won a free prize. Click here to claim your reward immediately.",
    work: "Please attend the project meeting tomorrow at 10 AM and review the attached project report.",
    personal: "Hey, are you free for dinner tonight? Let me know when you are available."
};

const COLORS = { Spam: "#dc2626", Work: "#2563eb", Personal: "#16a34a" };

const emailInput = document.getElementById("emailText");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");
const predictBtn = document.getElementById("predictBtn");

emailInput.addEventListener("input", () => {
    document.getElementById("charCount").textContent = emailInput.value.length;
});

emailInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) predictEmail();
});

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
    resultBox.classList.add("hidden");
}

async function predictEmail() {
    const emailText = emailInput.value.trim();
    errorBox.classList.add("hidden");

    if (!emailText) {
        showError("Please enter an email first.");
        return;
    }

    predictBtn.disabled = true;
    predictBtn.textContent = "Classifying...";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email: emailText })
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            showError(data.error || "Request failed.");
            return;
        }

        renderResult(data);
    } catch (error) {
        console.error(error);
        showError("Could not reach the server. Is the Flask app running?");
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = "Classify Email";
    }
}

function renderResult(data) {
    const percent = Math.round(data.confidence * 100);

    resultBox.className = "result " + data.category.toLowerCase();
    document.getElementById("category").textContent = data.category;
    document.getElementById("confidence").textContent = percent;
    document.getElementById("meterFill").style.width = percent + "%";

    const breakdown = document.getElementById("breakdown");
    breakdown.innerHTML = "";

    Object.entries(data.probabilities)
        .sort((a, b) => b[1] - a[1])
        .forEach(([label, prob]) => {
            const pct = (prob * 100).toFixed(1);
            const row = document.createElement("div");
            row.className = "break-row";
            row.innerHTML =
                '<span class="break-label"></span>' +
                '<div class="break-bar"><span></span></div>' +
                '<span class="break-val"></span>';
            row.querySelector(".break-label").textContent = label;
            row.querySelector(".break-val").textContent = pct + "%";
            const bar = row.querySelector(".break-bar span");
            bar.style.width = pct + "%";
            bar.style.background = COLORS[label] || "#64748b";
            breakdown.appendChild(row);
        });
}

function clearEmail() {
    emailInput.value = "";
    document.getElementById("charCount").textContent = "0";
    resultBox.classList.add("hidden");
    errorBox.classList.add("hidden");
    emailInput.focus();
}

function useExample(type) {
    emailInput.value = EXAMPLES[type];
    document.getElementById("charCount").textContent = emailInput.value.length;
    predictEmail();
}
