// Get elements
const addButton = document.getElementById("addButton");
const additionalInputs = document.getElementById("additionalInputs");
const dynamicInputContainer = document.getElementById("dynamicInputContainer");
const dynamicInput = document.getElementById("dynamicInput");
const fileInput = document.getElementById("fileInput");
const sendButton = document.getElementById("sendButton");
const questionInput = document.getElementById("questionInput");
const buttonsvg = document.querySelector(".buttonsvg");
const sharedInputContainer = document.getElementById("sharedInputContainer");
const clearChatsButton = document.getElementById("clearChatsButton");
const newChatButton = document.querySelector(".left button"); // Target "New Chat" button

// For displaying chat content
const question1 = document.getElementById("question1");
const question2 = document.getElementById("question2");
const solution = document.getElementById("solution");
const sourcesContainer = document.createElement("div"); // Container for sources
sourcesContainer.id = "sourcesContainer";
solution.parentElement.appendChild(sourcesContainer);

// Variables
let currentButton = null;
let uploadedFilePaths = []; // To store uploaded file paths globally


// Ensure the shared input container is visible across screens
function toggleScreenVisibility(screenToShow, screenToHide) {
    screenToShow.style.display = "flex";  // Show the screen - was "block"
    screenToHide.style.display = "none";
}

// Toggle the arc of circular buttons
addButton.addEventListener("click", () => {
    if (additionalInputs.classList.contains("show")) {
        additionalInputs.classList.remove("show");
        dynamicInputContainer.classList.add("hidden");
        currentButton = null;
    } else {
        additionalInputs.classList.add("show");
    }
});

// Add click listeners to the circular buttons
document.querySelectorAll(".circle-button").forEach(button => {
    button.addEventListener("click", () => {
        const type = button.getAttribute("data-type");

        // Hide previous input if any
        dynamicInputContainer.classList.add("hidden");
        dynamicInput.value = "";
        fileInput.value = "";

        // Position the input container just above the clicked button
        const buttonRect = button.getBoundingClientRect();
        const containerRect = buttonsvg.getBoundingClientRect();
        const relativeTop = buttonRect.top - containerRect.top;
        const relativeLeft = buttonRect.left - containerRect.left;

        dynamicInputContainer.style.top = `${relativeTop - 60}px`;
        dynamicInputContainer.style.left = `${relativeLeft - 75}px`;

        // Handle "Upload Document" with file input
        if (type === "Upload Document") {
          // Programmatically trigger file input click
            fileInput.click();
            dynamicInput.classList.add("hidden"); // Hide text input
        } else {
            dynamicInput.classList.remove("hidden");
            fileInput.classList.add("hidden");
            dynamicInput.placeholder = `Enter ${type}`;
        }

        dynamicInputContainer.classList.remove("hidden");
        currentButton = button;
    });
});

let uploadedFiles = []; // Store selected files

// // Handle file uploads (Extend the existing logic)
// fileInput.addEventListener("change", async () => {
//   const files = Array.from(fileInput.files);
//   if (files.length > 0) {
//       console.log(`Files selected: ${files.map(file => file.name).join(", ")}`);
//       alert(`Files selected: ${files.map(file => file.name).join(", ")}`);

//       // Process each selected file
//       for (const file of files) {
//           const uploadFormData = new FormData();
//           uploadFormData.append("file", file);

//           try {
//               // Send file to the backend
//               const response = await fetch("/api", {
//                   method: "POST",
//                   body: uploadFormData,
//               });

//               const result = await response.json();
//               if (response.ok) {
//                   alert(`File uploaded successfully: ${file.name}`);
//               } else {
//                   alert(`Error uploading ${file.name}: ${result.error}`);
//               }
//           } catch (error) {
//               console.error(`Error uploading ${file.name}:`, error);
//               alert(`An error occurred while uploading ${file.name}.`);
//           }
//       }

//       // Reset file input after processing
//       fileInput.value = "";
//       dynamicInputContainer.classList.add("hidden");
//   } else {
//       alert("No file selected.");
//   }
// });



// Handle file uploads
// Handle file uploads (separate from the question submission)
fileInput.addEventListener("change", async () => {
  const files = Array.from(fileInput.files);
  if (files.length > 0) {
      console.log(`Files selected: ${files.map(file => file.name).join(", ")}`);
      alert(`Files selected: ${files.map(file => file.name).join(", ")}`);

      const formData = new FormData();
      files.forEach((file) => formData.append("file", file));

      try {
          const response = await fetch("/api/upload", { // Separate endpoint for uploads
              method: "POST",
              body: formData,
          });

          if (response.ok) {
              const result = await response.json();
              uploadedFilePaths = result.file_paths; // Save file paths for later use
              console.log(`Files uploaded: ${uploadedFilePaths}`);
              alert("Files uploaded successfully.");
          } else {
              const result = await response.json();
              alert(`Error uploading files: ${result.error}`);
          }
      } catch (error) {
          console.error("Error uploading files:", error);
          alert("An error occurred while uploading the files.");
      }

      // Clear file input after processing
      fileInput.value = "";
  } else {
      alert("No file selected.");
  }
});



// Function to post data to the server
async function postData(url = '', data = {}) {
    const response = await fetch(url, {
        method: 'POST',
        body: data
    });
    return response.json();
}

// Function to make sources clickable
function formatSources(sources) {
  return sources.map(source => {
      // Check if it's a URL or a file path and format as clickable
      const isURL = source.startsWith("http://") || source.startsWith("https://");
      return isURL
          ? `<a href="${source}" target="_blank" rel="noopener noreferrer">${source}</a>`
          : `<a href="${source}" download>${source}</a>`;
  }).join("<br>");
}


// Send question to Flask backend
sendButton.addEventListener("click", async () => {
  if (!questionInput.value.trim()) {
      alert("Please enter a question.");
      return;
  }

  const formData = new FormData();
  formData.append("question", questionInput.value);

  // Check if there are additional inputs
  if (currentButton) {
      const type = currentButton.getAttribute("data-type");
      if (type && type !== "Upload Document") {
          formData.append(type, dynamicInput.value); // Add dynamic input value
      }
  }

  // Attach file paths for uploaded documents
  // const uploadedFilePaths = []; // Simulate receiving file paths from the /api/upload response
  // const response = await fetch("/api/upload", { method: "POST", body: formData });
  // files.forEach((file) => formData.append("file", file));

  if (uploadedFilePaths.length > 0) {
    formData.append("file_paths", JSON.stringify(uploadedFilePaths));
  }

  try {
      // Send data to the Flask backend
      const response = await fetch("/api", {
          method: "POST",
          body: formData,
      });

      const result = await response.json();

      if (response.ok) {
          toggleScreenVisibility(document.querySelector(".right2"), document.querySelector(".right1"));
          question1.innerHTML = questionInput.value; // Display user question
          question2.innerHTML = questionInput.value;
          solution.innerHTML = result.answer;

          if (result.sources) {
              sourcesContainer.innerHTML = `<strong>Sources:</strong><br>${formatSources(result.sources)}`;
          } else {
              sourcesContainer.innerHTML = "";
          }
      } else {
          alert(`Error: ${result.error}`);
      }
  } catch (error) {
      console.error("Error sending question:", error);
      alert("An error occurred while processing your request.");
  }

  // Clear inputs after submission
  questionInput.value = "";
  dynamicInput.value = "";
  fileInput.value = "";
  dynamicInputContainer.classList.add("hidden");
  additionalInputs.classList.remove("show");
  currentButton = null;
  uploadedFilePaths = []; // Reset file paths after submission
});



// Submit query on Enter key press
questionInput.addEventListener("keypress", (event) => {
  if (event.key === "Enter") {
      sendButton.click();
      event.preventDefault();
  }
});


// Adjust the height of the text input dynamically
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto"; // Reset height to calculate new height
  questionInput.style.height = `${questionInput.scrollHeight}px`; // Set the height dynamically
});


// Clear chat history
clearChatsButton.addEventListener("click", async () => {
    try {
        const response = await fetch("/clear_chats", {
            method: "POST",
        });
        const result = await response.json();
        if (response.ok) {
            alert("Chat history cleared successfully.");
            // Refresh the page to reflect the changes
            window.location.reload();
        } else {
            alert(`Error: ${result.error}`);
        }
    } catch (error) {
        console.error("Error clearing chat history:", error);
        alert("An error occurred while clearing chat history.");
    }
});


// Handle "New Chat" functionality
newChatButton.addEventListener("click", () => {
    // Clear the input field
    questionInput.value = "";

    // Hide the right2 (chat interface)
    const right2 = document.querySelector(".right2");
    const right1 = document.querySelector(".right1");

    right1.style.display = "block"; // Show the initial "CustomGPT" view
    right2.style.display = "none"; // Hide the chat content view

    // Clear any displayed chat history or solution
    question1.innerHTML = "";
    question2.innerHTML = "";
    solution.innerHTML = "";
    sourcesContainer.innerHTML = "";

    alert("New chat started!");
});


// Ensure shared input container adapts to screen changes
function ensureSharedContainerVisibility() {
  sharedInputContainer.style.display = "flex";
}
window.addEventListener("resize", ensureSharedContainerVisibility);

// Initial setup
ensureSharedContainerVisibility();