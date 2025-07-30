document.getElementById("toggle-chat").addEventListener("click", function () {
  const chatbox = document.getElementById("chatbox");
  chatbox.style.display =
    chatbox.style.display === "none" || chatbox.style.display === ""
      ? "block"
      : "none";
  // Focus the input when chatbox is shown
  if (chatbox.style.display === "block") {
    const messageInput = document.getElementById("message-input");
    if (messageInput) messageInput.focus();
  }
});

document.getElementById("close-chat").addEventListener("click", function () {
  const popup = document.querySelector(".clear-popup");
  popup.style.display = "block";
});

document.querySelector(".cancel-btn").addEventListener("click", function () {
  const popup = document.querySelector(".clear-popup");
  popup.style.display = "none";
});

document.querySelector(".clear-btn").addEventListener("click", function () {
  clearChatMessages();
  const popup = document.querySelector(".clear-popup");
  popup.style.display = "none";
});

function clearChatMessages() {
  const messages = document.getElementById("messages");
  if (messages) {
    let child = messages.firstChild;
    while (child) {
      if (
        !child.classList ||
        (!child.classList.contains("") &&
          !child.classList.contains("menu-container"))
      ) {
        const nextChild = child.nextSibling;
        messages.removeChild(child);
        child = nextChild;
      } else {
        child = child.nextSibling;
      }
    }
  }
}

document.getElementById("minimize-chat").addEventListener("click", function () {
  document.getElementById("chatbox").style.display = "none";
});

document.getElementById("send-button").addEventListener("click", function () {
  sendMessage();
});

document
  .getElementById("message-input")
  .addEventListener("keypress", function (event) {
    if (event.key === "Enter") {
      sendMessage();
    }
  });

function addMessage({ content, type, messageType = "responses" }) {
  const messageContainer = document.getElementById("messages");
  const messageDiv = document.createElement("div");

  if (messageType === "input") {
    messageDiv.className = `message ${type}`;
  } else if (messageType === "dropdown") {
    messageDiv.className = `message-dropdown ${type}`;
  } else {
    messageDiv.className = `message-daq ${type}`;
  }

  if (content.startsWith("<") && content.endsWith(">")) {
    messageDiv.innerHTML = content;
  } else {
    messageDiv.textContent = content;
  }

  const loader = document.getElementById("message-loader");
  if (loader && loader.parentNode === messageContainer) {
    messageContainer.insertBefore(messageDiv, loader);
  } else {
    messageContainer.appendChild(messageDiv);
  }
  if (messageType === "input" || messageType === "dropdown") {
    setTimeout(() => {
      messageContainer.scrollTop = messageContainer.scrollHeight;
    }, 0);
  }
}

function addMessageLinks(links, type) {
  if (
    links &&
    (typeof links === "string" ||
      typeof links === "null" ||
      (Array.isArray(links) && links.length > 0))
  ) {
    const messageContainer = document.getElementById("messages");
    const loader = document.getElementById("message-loader");

    const messageDiv = document.createElement("div");
    messageDiv.className = `message-daq ${type}`;

    let linkContent = `<strong>Read more about the product:</strong><ul>`;
    if (typeof links === "string") {
      linkContent += `<li>${links}<img src="./static/images/external.png" alt="Baumalight Mascot" class="link-icon-sm"></li>`;
    } else if (Array.isArray(links)) {
      links.forEach((link) => {
        linkContent += `<li><a>${link}</a><img src="./static/images/external.png" alt="Baumalight Mascot" class="link-icon-sm"></li>`;
      });
    }
    linkContent += `</ul>`;
    messageDiv.innerHTML = linkContent;

    if (loader && loader.parentNode === messageContainer) {
      messageContainer.insertBefore(messageDiv, loader);
    } else {
      messageContainer.appendChild(messageDiv);
    }
    // Removed scrollTop logic
  }
}

function addMessageDescription(message, type) {
  if (message && message.trim()) {
    const messageContainer = document.getElementById("messages");
    const loader = document.getElementById("message-loader");
    const messageDiv = document.createElement("div");
    messageDiv.className = `message-daq ${type}`;

    // Limit message to 100 words
    const words = message.trim().split(/\s+/);
    const limitedMessage =
      words.length > 180 ? words.slice(0, 180).join(" ") + "..." : message;

    messageDiv.innerHTML = `<p>${limitedMessage}</p>`;
    if (loader && loader.parentNode === messageContainer) {
      messageContainer.insertBefore(messageDiv, loader);
    } else {
      messageContainer.appendChild(messageDiv);
    }
    messageContainer.scrollTop += 10;
    // Removed scroll-to-bottom for answers as per user request
  }
}

function addMessageQuestion(questions, type) {
  if (questions && Array.isArray(questions) && questions.length > 0) {
    const messageContainer = document.getElementById("messages");

    const messageDiv = document.createElement("div");
    messageDiv.className = `message-daq ${type}`;

    questions.forEach((question) => {
      const button = document.createElement("button");
      button.className = "clickable-question statement-container1";
      button.setAttribute("data-question", question.trim());
      button.textContent = question.trim();
      messageDiv.appendChild(button);
    });

    const loader = document.getElementById("message-loader");
    if (loader && loader.parentNode === messageContainer) {
      messageContainer.insertBefore(messageDiv, loader);
    } else {
      messageContainer.appendChild(messageDiv);
    }
    messageContainer.scrollTop += 10;
  }
}

function addMessageDefault() {
  const messageContainer = document.getElementById("messages");

  const messageDiv = document.createElement("div");
  messageDiv.className = "message-daq";

  messageDiv.innerHTML = `
        <p>Sorry! couldn't find what you were searching for. here are some options to try from</p>
        <strong>Related about the product:</strong>
        <button class="clickable-question statement-container1" data-question="would you like to know more about our tractor models?">would you like to know more about our tractor models?</button>
        <button class="clickable-question statement-container1" data-question="what products are eligible for the factory discount program?">what products are eligible for the factory discount program?</button>
        <button class="clickable-question statement-container1" data-question="Would you prefer leasing or purchasing with financing options?">Would you prefer leasing or purchasing with financing options?</button>
    `;

  const loader = document.getElementById("message-loader");
  if (loader && loader.parentNode === messageContainer) {
    messageContainer.insertBefore(messageDiv, loader);
  } else {
    messageContainer.appendChild(messageDiv);
  }
  messageContainer.scrollTop += 10;
  // Removed scrollTop logic
}

function sendMessage(event) {
  const input = document.getElementById("message-input");
  if (event && event.target !== input) return;

  const messageText = input.value.trim();
  const messageContainer = document.getElementById("messages");

  if (messageText) {
    addMessage(messageText, "sender", "input");
    input.value = "";
    input.scrollTop = 0;
    input.setSelectionRange(0, 0);
    input.rows = 1;
    input.style.height = "55px";
    setTimeout(() => {
      input.value = "";
      input.scrollTop = 0;
      input.setSelectionRange(0, 0);
      input.rows = 1;
      input.style.height = "55px";
      input.focus();
      console.log(
        "After clear (timeout):",
        JSON.stringify(input.value),
        "Length:",
        input.value.length
      );
    }, 10);
    input.focus();
    console.log(
      "After clear:",
      JSON.stringify(input.value),
      "Length:",
      input.value.length
    );
    showLoader();

    const processedMessage = getChatbotResponse(messageText);

    fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: messageText,
        max_results: 3,
        include_sources: true,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        hideLoader();

        if (data.response) {
          try {
            const response = JSON.parse(data.response);
            if (response.Description)
              addMessageDescription(response.Description, "receiver");
            if (response.Links) addMessageLinks(response.Links, "receiver");
            if (response.Questions)
              addMessageQuestion(response.Questions, "receiver");
          } catch (error) {
            console.error("Error parsing response:", error);
            addBotResponse(data.response, messageContainer); // Changed to addBotResponse
          }
        } else {
          addMessageDefault("receiver");
        }

        // Remove the timestamp addition here
        // Create a container for timestamp and feedback icons
        const timestampContainer = document.createElement("div");
        timestampContainer.className = "timestamp-feedback-container";

        // Add timestamp
        const timestampElement = document.createElement("div");
        timestampElement.className = "message-timestamp timestamp-left";
        timestampElement.textContent = getCurrentTime();

        // Add feedback icons
        const feedbackIcons = document.createElement("div");
        feedbackIcons.className = "feedback-icons";
        feedbackIcons.innerHTML = `
                     <i class="fa-regular fa-thumbs-up thumb-icon" id="thumbs-up"></i>
                     <i class="fa-regular fa-thumbs-down thumb-icon" id="thumbs-down"></i>
                 `;

        // Append timestamp and feedback icons to their container
        timestampContainer.appendChild(timestampElement);
        timestampContainer.appendChild(feedbackIcons);

        // Append the container to the message container
        const loader = document.getElementById("message-loader");
        if (loader && loader.parentNode === messageContainer) {
          messageContainer.insertBefore(timestampContainer, loader);
        } else {
          messageContainer.appendChild(timestampContainer);
        }

        // Removed scrollTop logic
      })
      .catch((error) => {
        console.error("Error:", error);
        addMessageDefault("receiver");
        hideLoader();
      });
  }
}

function addBotResponse(content, messageContainer) {
  const botResponse = document.createElement("div");
  botResponse.className = "message-container bot-message";

  const botMessage = document.createElement("div");
  botMessage.className = "message-daq receiver";
  botMessage.innerHTML = content;

  botResponse.appendChild(botMessage);

  const loader = document.getElementById("message-loader");
  if (loader && loader.parentNode === messageContainer) {
    messageContainer.insertBefore(botResponse, loader);
  } else {
    messageContainer.appendChild(botResponse);
  }
}

function isJSON(str) {
  try {
    JSON.parse(str);
    return true;
  } catch (error) {
    return false;
  }
}

function printInputMessage(message) {
  console.log("Input Message:", message);
}

function sendInputMessage() {
  const input = document.getElementById("message-input");
  const messageText = input.value.trim();

  if (messageText) {
    addMessage(messageText, true); // Add user message with timestamp
    input.value = "";
    input.scrollTop = 0;
    input.setSelectionRange(0, 0);
    input.rows = 1;
    input.style.height = "55px";
    setTimeout(() => {
      input.value = "";
      input.scrollTop = 0;
      input.setSelectionRange(0, 0);
      input.rows = 1;
      input.style.height = "55px";
      input.focus();
      console.log(
        "After clear (timeout):",
        JSON.stringify(input.value),
        "Length:",
        input.value.length
      );
    }, 10);
    input.focus();
    console.log(
      "After clear:",
      JSON.stringify(input.value),
      "Length:",
      input.value.length
    );
    showLoader();

    fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        prompt: messageText,
        max_results: 3,
        include_sources: true,
      }),
    })
      .then((response) => response.json())
      .then((data) => {
        hideLoader();
        if (data.response) {
          try {
            const response = JSON.parse(data.response);
            if (response.Description)
              addMessageDescription(response.Description, "receiver");
            if (response.Links) addMessageLinks(response.Links, "receiver");
            if (response.Questions)
              addMessageQuestion(response.Questions, "receiver");
          } catch (error) {
            console.error("Error parsing response:", error);
            addMessage(data.response, false); // Add bot message with timestamp
          }
        } else {
          addMessageDefault("receiver");
        }
      })
      .catch((error) => {
        console.error("Error:", error);
        addMessageDefault("receiver");
        hideLoader();
      });
  }
}

document
  .getElementById("send-button")
  .addEventListener("click", sendInputMessage);

document.addEventListener("DOMContentLoaded", function () {
  const messageInput = document.getElementById("message-input");
  const sendButton = document.getElementById("send-button");

  messageInput.addEventListener("input", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendInputMessage();
    }
  });

  sendButton.addEventListener("click", function () {
    const message = messageInput.value.trim();
    if (message) {
      sendInputMessage();
    }
  });
});

function getChatbotResponse(query) {
  let response = "";
  return response;
}

document.addEventListener("DOMContentLoaded", function () {
  const messageInput = document.getElementById("message-input");
  const sendButton = document.getElementById("send-button");
  const chatContainer = document.querySelector(".chatbox-messages");

  function sendMessage(text) {
    messageInput.value = text;

    const inputEvent = new Event("input", { bubbles: true });
    messageInput.dispatchEvent(inputEvent);

    const clickEvent = new MouseEvent("click", {
      bubbles: true,
      cancelable: true,
      view: window,
    });
    sendButton.dispatchEvent(clickEvent);
  }

  function activateQuestionListeners() {
    chatContainer.addEventListener("click", function (e) {
      const clickedQuestion = e.target.closest(".clickable-question");
      if (clickedQuestion) {
        const questionText =
          clickedQuestion.dataset.question || clickedQuestion.textContent;
        sendMessage(questionText.trim());
      }
    });
  }

  activateQuestionListeners();
});

document.addEventListener("DOMContentLoaded", function () {
  const messagesContainer = document.getElementById("messages");

  chatbox.addEventListener("click", function (event) {
    if (event.target.classList.contains("thumb-icon")) {
      const iconType = event.target.id;
      const feedbackContainer = event.target.closest(".feedback-icons");

      //   if (feedbackContainer) {
      //     // Check if feedback message already exists
      //     const existingFeedback = document.querySelector(".message-daqr");
      //     if (existingFeedback) {
      //       // If feedback already exists, don't add another one
      //       return;
      //     }

      //     const feedbackMessage = document.createElement("div");
      //     feedbackMessage.className = "message-daqr";
      //     feedbackMessage.textContent =
      //       iconType === "thumbs-up"
      //         ? "Thank you for your feedback!"
      //         : "Sorry to hear that! We value your feedback.";

      //     const messageContainer = document.getElementById("messages");
      //     messageContainer.appendChild(feedbackMessage);

      //     // Remove all feedback icons after clicking
      //     removeFeedbackIcons();

      //     messageContainer.scrollTop = messageContainer.scrollHeight;
      //   }
    }
  });
});

function showLoader() {
  const loader = document.getElementById("message-loader");
  loader.style.display = "block";
}

function hideLoader() {
  const loader = document.getElementById("message-loader");
  loader.style.display = "none";
}

document.addEventListener("DOMContentLoaded", function () {
  const initialMessageContainers = document.querySelectorAll(
    ".message-initial-input"
  );
  const initialMessages = document.querySelectorAll(".message-initial");
  const chatbox = document.getElementById("chatbox");

  initialMessages.forEach((message) => {
    message.addEventListener("click", function () {
      const messageText = this.textContent.trim();
      const input = document.getElementById("message-input");
      input.value = messageText;
      input.dispatchEvent(new Event("input", { bubbles: true }));
      document.getElementById("send-button").click();
      initialMessageContainers.forEach((container) => {
        container.style.display = "none";
      });
    });
  });
});

document.addEventListener("DOMContentLoaded", function () {
  document
    .getElementById("chatbox")
    .addEventListener("click", function (event) {
      if (event.target.classList.contains("thumb-icon")) {
        const iconType = event.target.id;
        if (iconType === "thumbs-up") {
        } else if (iconType === "thumbs-down") {
        }
        event.target.style.opacity = 0.5;
      }
    });
});

function toggleDropdown(dropdownId) {
  const dropdown = document.getElementById(dropdownId);
  const chevron = dropdown.previousElementSibling.querySelector(".chevron img");

  if (dropdown.style.display === "grid" || dropdown.style.display === "") {
    dropdown.style.display = "none";
    chevron.style.transform = "rotate(180deg)";
  } else {
    dropdown.style.display = "grid";
    chevron.style.transform = "rotate(0deg)";
  }
}

document.querySelectorAll(".menu-item").forEach((button) => {
  button.addEventListener("click", () => {
    button.classList.toggle("active");
    const dropdown = button.nextElementSibling;
    if (dropdown.style.maxHeight) {
      dropdown.style.maxHeight = null;
    } else {
      dropdown.style.maxHeight = dropdown.scrollHeight + "px";
    }
  });
});

function addMessage(message, type, messageType = "responses") {
  const messageContainer = document.getElementById("messages");
  const messageDiv = document.createElement("div");

  if (messageType === "input") {
    messageDiv.className = `message ${type}`;
  } else if (messageType === "dropdown") {
    messageDiv.className = `message-dropdown ${type}`;
  } else {
    messageDiv.className = `message-daq ${type}`;
  }

  if (content.startsWith("<") && content.endsWith(">")) {
    messageDiv.innerHTML = content;
  } else {
    messageDiv.textContent = content;
  }

  const loader = document.getElementById("message-loader");
  if (loader && loader.parentNode === messageContainer) {
    messageContainer.insertBefore(messageDiv, loader);
  } else {
    messageContainer.appendChild(messageDiv);
  }
  if (messageType === "input" || messageType === "dropdown") {
    // Removed scrollTop logic
  }
}

function handleMessage(message, isUserMessage) {
  const messageContainer = document.getElementById("messageContainer");

  if (isUserMessage) {
    addUserMessage(message, messageContainer);
  } else {
    addBotResponse(message, messageContainer);
  }

  // Removed scrollTop logic
}

function addUserMessage(content, messageContainer) {
  const messageWrapper = document.createElement("div");
  messageWrapper.className = "message-container user-message";

  const messageDiv = document.createElement("div");
  messageDiv.className = "message sender";
  messageDiv.textContent = content;

  const timestampElement = document.createElement("div");
  timestampElement.className = "message-timestamp timestamp-right";
  timestampElement.textContent = getCurrentTime();

  messageWrapper.appendChild(messageDiv);
  messageWrapper.appendChild(timestampElement);

  const loader = document.getElementById("message-loader");
  if (loader && loader.parentNode === messageContainer) {
    messageContainer.insertBefore(messageWrapper, loader);
  } else {
    messageContainer.appendChild(messageWrapper);
  }
  setTimeout(() => {
    messageContainer.scrollTop = messageContainer.scrollHeight;
  }, 0);
}

// Define static responses for dropdown items
const dropdownResponses = {
  Generator: {
    description:
      "Baumalight's TX, KR and QC series of PTO tractor generators give you a variety of choices – from running small tools and appliances to powering large motors on equipment such as silo unloaders and milking systems.",
    links: ["https://online-order.baumalight.com/product/generators-en"],
  },
  "PTO Generators": {
    description:
      "PTO generators, power is as close by as your tractor. A portable, convenient and reliable solution for emergency back-up, in-field repairs or remote power for un-serviced locations",
    links: ["https://online-order.baumalight.com/product/generators-en"],
  },
  "TX Series": {
    description:
      "Baumalight TX series of PTO generators, power is as close by as your tractor. With one less engine to maintain on your property, the TX series of portable generators gives you the advantage of using your reliable tractor during an emergency situation or simply for additional power when required.",
    links: ["https://online-order.baumalight.com/product/generators/tx-models"],
  },
  "KR Series": {
    description:
      "KR series generators are designed to fit tractors from 45 to 97 horsepower and support power requirements from 30 kilowatts to 65 kilowatts. The KR models generator assembly is made in Canada and the main components are made in Europe.",
    links: ["https://online-order.baumalight.com/product/generators/kr-models"],
  },
  "QC Series": {
    description:
      "QC series tractor powered generator is ideal for your farm or rural property. At 100% load, these QC PTO models offer 12 KW to 100 KW and momentary surge from 25 up to 300 KW. ",
    links: [
      "https://online-order.baumalight.com/product/generators/qc-singlephase",
    ],
  },
  "120/240 Single Phase": {
    description:
      "120/240 Single Phase. Designed for mobile or standby electrical power, the QC series tractor powered generator is ideal for your farm or rural property. At 100% load, these QC PTO models offer 12 KW to 100 KW and momentary surge from 25 up to 300 KW.",
    links: [
      "https://online-order.baumalight.com/product/generators/qc-singlephase",
    ],
  },
  "120/208 Volt 3-Phase": {
    description:
      "Baumalight offers QC generators that can be configured for 120/208 voltage and features 30 KW to 105 KW at 100% load and a momentary surge from 85 KW to 315 KW. ",
    links: ["https://online-order.baumalight.com/product/generators/qc-120"],
  },
  "120/240 Volt 3-Phase": {
    description:
      "QC PTO generator that is configured for 120/240 three phase voltage, at 100% load these generators offer 30 KW to 105 KW and a momentary surge from 85 to 315 KW.",
    links: ["https://online-order.baumalight.com/product/generators/qc-3phase"],
  },
  "480 Volt 3-Phase": {
    description:
      "QC PTO generator for 480 volt three phase models are ideal for your farm or rural property. At 100% load, these QC PTO models offer 33 KW to 113 KW and momentary surge from 85 KW up to 315 KW.",
    links: [
      "https://online-order.baumalight.com/product/generators/qc-volt480",
    ],
  },
  "600 Volt 3-Phase": {
    description:
      "QC PTO tractor-driven generator models for a Dedicated 377/600 volt three phase output, which is the most frequent voltage distribution feed in Canada and also sometimes referred to as 550 volt. At 100% load, these QC PTO models offer 27 KW to 100 KW and momentary surge from 45 to 300 KW.",
    links: [
      "https://online-order.baumalight.com/product/generators/qc-volt600",
    ],
  },
};

const messageContainer = document.getElementById("messages");

function addMessage(message, isUserMessage) {
  const messageContainer = document.getElementById("messages");
  const messageElement = document.createElement("div");
  messageElement.className = isUserMessage
    ? "message-container user-message"
    : "message-container bot-message";
  messageElement.innerHTML = `
        <div class="message ${
          isUserMessage ? "sender" : "receiver"
        }">${message}</div>
        <div class="message-timestamp ${
          isUserMessage ? "timestamp-right" : "timestamp-left"
        }">${getCurrentTime()}</div>
    `;
  const loader = document.getElementById("message-loader");
  if (loader && loader.parentNode === messageContainer) {
    messageContainer.insertBefore(messageElement, loader);
  } else {
    messageContainer.appendChild(messageElement);
  }
  setTimeout(() => {
    messageContainer.scrollTop = messageContainer.scrollHeight;
  }, 0);
}

// Keep your existing dropdownResponses object
function handleDropdownItemClick(event) {
  const dropdownItem = event.currentTarget;
  const dropdown = dropdownItem.closest(".dropdown-content");

  let itemText = Array.from(dropdownItem.childNodes)
    .filter((node) => node.nodeType === Node.TEXT_NODE)
    .map((node) => node.textContent.trim())
    .join("");

  itemText = itemText.trim();

  if (dropdown) {
    dropdown.classList.remove("active");
    dropdown.style.display = "none";
  }

  addDropdownMessage(itemText, "sender1", messageContainer);

  displayStaticResponse(itemText, messageContainer);

  // Ensure input is always focused after clicking a dropdown item
  const messageInput = document.getElementById("message-input");
  if (messageInput) messageInput.focus();
}

function addDropdownMessage(content, type, messageContainer) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message-dropdown ${type}`;
  messageDiv.textContent = content;
  const timestampElement = document.createElement("div");
  timestampElement.className = "message-timestamp timestamp-right";
  timestampElement.textContent = getCurrentTime();
  const loader = document.getElementById("message-loader");
  if (loader && loader.parentNode === messageContainer) {
    messageContainer.insertBefore(messageDiv, loader);
  } else {
    messageContainer.appendChild(messageDiv);
  }
  messageContainer.appendChild(timestampElement);
  setTimeout(() => {
    messageContainer.scrollTop = messageContainer.scrollHeight;
  }, 0);
}

function handleQuestionClick(event) {
  const button = event.target;
  if (button.classList.contains("clickable-question1")) {
    const question =
      button.getAttribute("data-question1") || button.textContent;

    // Add the clicked question as a sender1 message
    addMessageAsSender1(question, messageContainer);

    // Display the static response for this question
    displayStaticResponse(question, messageContainer);

    // Ensure input is always focused after clicking a question
    const messageInput = document.getElementById("message-input");
    if (messageInput) messageInput.focus();
  }
}

function addMessageAsSender1(content, messageContainer) {
  const messageDiv = document.createElement("div");
  messageDiv.className = "message-dropdown sender1";
  messageDiv.textContent = content;
  const timestampElement = document.createElement("div");
  timestampElement.className = "message-timestamp timestamp-right";
  timestampElement.textContent = getCurrentTime();
  const loader = document.getElementById("message-loader");
  if (loader && loader.parentNode === messageContainer) {
    messageContainer.insertBefore(messageDiv, loader);
  } else {
    messageContainer.appendChild(messageDiv);
  }
  messageContainer.appendChild(timestampElement);
  setTimeout(() => {
    messageContainer.scrollTop = messageContainer.scrollHeight;
  }, 0);
}

function displayStaticResponse(itemText, messageContainer) {
  const response = dropdownResponses[itemText];

  if (response) {
    // Create a container for the new response
    const responseContainer = document.createElement("div");

    // Add description
    if (response.description) {
      const descDiv = document.createElement("div");
      descDiv.className = "message-daq receiver1";
      descDiv.innerHTML = `<p>${response.description}</p>`;
      responseContainer.appendChild(descDiv);
    }

    // Add links
    if (response.links && response.links.length > 0) {
      const linksDiv = document.createElement("div");
      linksDiv.className = "message-daq receiver1";
      let linksHtml = "<strong>Read more about the product</strong><ul>";
      response.links.forEach((link) => {
        // Handle special cases for display text
        let displayText = itemText;
        if (itemText === "Upcoming New Product Release") {
          displayText = "Barrier Mower";
        } else if (itemText === "Discounts and Offers") {
          displayText = "Factory Discounts";
        }

        linksHtml += `<li><a href="${link}" target="_blank">${displayText}</a><img src="./static/images/external.png" alt="Baumalight Mascot" class="link-icon-sm"></li>`;
      });
      linksHtml += "</ul>";
      linksDiv.innerHTML = linksHtml;
      responseContainer.appendChild(linksDiv);
    }

    // Add questions.
    if (response.questions && response.questions.length > 0) {
      const questionsDiv = document.createElement("div");
      questionsDiv.className = "message-daq receiver1";
      let questionsHTML = "<strong>Below are the related Models</strong><ul>";
      questionsDiv.innerHTML = questionsHTML;

      response.questions.forEach((question) => {
        const button = document.createElement("button");
        button.className = "clickable-question1 statement-container1";
        button.setAttribute("data-question1", question);

        // Remove suffixes from display text while keeping the full value in data attribute
        let displayText = question
          .replace(/ Excvator$/, "")
          .replace(/ Tractor$/, "")
          .replace(/ Skidsteer$/, "")
          .replace(/ Skidsteers$/, "");

        button.textContent = displayText;
        questionsDiv.appendChild(button);
      });

      responseContainer.appendChild(questionsDiv);
    }

    // Append the new response container to the message container
    const loader = document.getElementById("message-loader");
    if (loader && loader.parentNode === messageContainer) {
      messageContainer.insertBefore(responseContainer, loader);
    } else {
      messageContainer.appendChild(responseContainer);
    }
    messageContainer.scrollTop += 10;

    // Create a container for timestamp and feedback icons
    const timestampContainer = document.createElement("div");
    timestampContainer.className = "timestamp-feedback-container";

    // Add timestamp
    const timestampElement = document.createElement("div");
    timestampElement.className = "message-timestamp timestamp-left";
    timestampElement.textContent = getCurrentTime();

    // Add feedback icons
    const feedbackIcons = document.createElement("div");
    feedbackIcons.className = "feedback-icons";
    feedbackIcons.innerHTML = `
            <i class="fa-regular fa-thumbs-up thumb-icon" id="thumbs-up"></i>
            <i class="fa-regular fa-thumbs-down thumb-icon" id="thumbs-down"></i>
        `;

    // Append timestamp and feedback icons to their container
    timestampContainer.appendChild(timestampElement);
    timestampContainer.appendChild(feedbackIcons);

    // Append the container to the message container
    if (loader && loader.parentNode === messageContainer) {
      messageContainer.insertBefore(timestampContainer, loader);
    } else {
      messageContainer.appendChild(timestampContainer);
    }

    // // Scroll to the bottom of the chat
    // messageContainer.scrollTop += 2; // This line is removed as per the edit hint
  } else {
    // If no response found, add a default message
    const defaultMessageDiv = document.createElement("div");
    defaultMessageDiv.className = "message-daq receiver1";
    defaultMessageDiv.innerHTML = `<p>I couldn't find any specific product in your request.</p>`;
    // Create a container for timestamp and feedback icons
    const timestampContainer = document.createElement("div");
    timestampContainer.className = "timestamp-feedback-container";

    // Add timestamp
    const timestampElement = document.createElement("div");
    timestampElement.className = "message-timestamp timestamp-left";
    timestampElement.textContent = getCurrentTime();

    // Add feedback icons
    const feedbackIcons = document.createElement("div");
    feedbackIcons.className = "feedback-icons";
    feedbackIcons.innerHTML = `
            <i class="fa-regular fa-thumbs-up thumb-icon" id="thumbs-up"></i>
            <i class="fa-regular fa-thumbs-down thumb-icon" id="thumbs-down"></i>
        `;

    // Append timestamp and feedback icons to their container
    timestampContainer.appendChild(timestampElement);
    timestampContainer.appendChild(feedbackIcons);

    // Append the container to the message container
    const loader = document.getElementById("message-loader");
    if (loader && loader.parentNode === messageContainer) {
      messageContainer.insertBefore(defaultMessageDiv, loader);
    } else {
      messageContainer.appendChild(defaultMessageDiv);
    }
    messageContainer.scrollTop += 10;
    messageContainer.appendChild(timestampContainer);
    // Removed scrollTop logic
  }
  // Removed scrollTop logic
}

// Modified DOMContentLoaded event listener
document.addEventListener("DOMContentLoaded", () => {
  showDropdown("whatsNewDropdown");

  // Add click listeners to dropdown items
  const dropdownItems = document.querySelectorAll(".dropdown-item");
  dropdownItems.forEach((item) => {
    item.addEventListener("click", handleDropdownItemClick);
  });

  // Add click listener for clickable questions in the chat
  messageContainer.addEventListener("click", handleQuestionClick);

  // Chat input and send button functionality remains the same
  const messageInput = document.getElementById("message-input");
  const sendButton = document.getElementById("send-button");

  sendButton.addEventListener("click", () => {
    const message = messageInput.value.trim();
    if (message) {
      addMessage(message, true);
      messageInput.value = "";
      messageInput.scrollTop = 0;
      messageInput.setSelectionRange(0, 0);
      messageInput.rows = 1;
      messageInput.style.height = "55px";
      messageInput.focus();
      console.log(
        "After clear:",
        JSON.stringify(messageInput.value),
        "Length:",
        messageInput.value.length
      );
      // Your existing chat handling logic here
    }
  });

  messageInput.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
      sendButton.click();
    }
  });
});

// Function to remove all feedback icons
function removeFeedbackIcons() {
  const feedbackIcons = document.querySelectorAll(".feedback-icons");
  feedbackIcons.forEach((icons) => icons.remove());
}

// Update the click handler for feedback icons
document.addEventListener("click", function (event) {
  if (event.target.classList.contains("thumb-icon")) {
    const iconType = event.target.id;
    const feedbackContainer = event.target.closest(".feedback-icons");

    if (feedbackContainer) {
      const feedbackMessage = document.createElement("div");
      feedbackMessage.className = "message-daqr";
      feedbackMessage.textContent =
        iconType === "thumbs-up"
          ? "Thank you for your feedback!"
          : "Sorry to hear that! We value your feedback.";

      const messageContainer = document.getElementById("messages");
      const loader = document.getElementById("message-loader");
      if (loader && loader.parentNode === messageContainer) {
        messageContainer.insertBefore(feedbackMessage, loader);
      } else {
        messageContainer.appendChild(feedbackMessage);
      }

      // Remove all feedback icons after clicking
      removeFeedbackIcons();

      // messageContainer.scrollTop += 2; // This line is removed as per the edit hint
    }
  }
});

function toggleDropdown(dropdownId) {
  const dropdown = document.getElementById(dropdownId);
  const allDropdowns = document.querySelectorAll(".dropdown-content");

  // Close all other dropdowns
  allDropdowns.forEach((d) => {
    if (d.id !== dropdownId) {
      d.classList.remove("active");
      d.style.display = "none";
    }
  });

  // Toggle the clicked dropdown
  if (dropdown) {
    const isCurrentlyActive = dropdown.classList.contains("active");
    dropdown.classList.toggle("active");
    dropdown.style.display = isCurrentlyActive ? "none" : "grid";

    // Toggle the chevron rotation
    const button = document.querySelector(
      `button[onclick="toggleDropdown('${dropdownId}')"]`
    );
    if (button) {
      const chevron = button.querySelector(".chevron img");
      if (chevron) {
        chevron.style.transform = isCurrentlyActive
          ? "rotate(180deg)"
          : "rotate(0deg)";
      }
    }
  }
}

// Modify the DOMContentLoaded event listener
document.addEventListener("DOMContentLoaded", () => {
  // Show the What's New dropdown initially if needed
  showDropdown("whatsNewDropdown");

  // Add click listeners to dropdown items
  const dropdownItems = document.querySelectorAll(".dropdown-item");
  dropdownItems.forEach((item) => {
    item.addEventListener("click", handleDropdownItemClick);
  });

  // Chat input and send button functionality
  const messageInput = document.getElementById("message-input");
  const sendButton = document.getElementById("send-button");

  sendButton.addEventListener("click", () => {
    const message = messageInput.value.trim();
    if (message) {
      addMessage(message, true);
      messageInput.value = "";
      messageInput.scrollTop = 0;
      messageInput.setSelectionRange(0, 0);
      messageInput.rows = 1;
      messageInput.style.height = "55px";
      messageInput.focus();
      console.log(
        "After clear:",
        JSON.stringify(messageInput.value),
        "Length:",
        messageInput.value.length
      );
      // Your existing chat handling logic here
    }
  });

  // Add keypress event for Enter key
  messageInput.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
      sendButton.click();
    }
  });
});

// Keep your existing showDropdown function
function showDropdown(dropdownId) {
  const dropdown = document.getElementById(dropdownId);
  if (dropdown) {
    dropdown.classList.add("active");
    dropdown.style.display = "grid";
  }
}

// Handle send button click
document.getElementById("send-button").addEventListener("click", () => {
  const messageInput = document.getElementById("message-input");
  const message = messageInput.value.trim();

  if (message) {
    addMessage(message, true);
    messageInput.value = "";
    messageInput.scrollTop = 0;
    messageInput.setSelectionRange(0, 0);
    messageInput.rows = 1;
    messageInput.style.height = "55px";
    messageInput.focus(); // Ensure input is always focused after sending
    console.log(
      "After clear:",
      JSON.stringify(messageInput.value),
      "Length:",
      messageInput.value.length
    );

    // Simulate bot response (replace with your actual bot response logic)
    setTimeout(() => {
      addMessage(`Response to: ${message}`, false);
      // Ensure input is always focused after bot response
      messageInput.focus();
    }, 1000);
  }
});

// Function to get current time
function getCurrentTime() {
  const now = new Date();
  let hours = now.getHours();
  let minutes = now.getMinutes();
  const ampm = hours >= 12 ? "PM" : "AM";

  // Convert to 12-hour format
  hours = hours % 12;
  hours = hours ? hours : 12; // handle midnight (0 hours)

  // Add leading zero to minutes if needed
  minutes = minutes < 10 ? "0" + minutes : minutes;

  return `${hours}:${minutes} ${ampm}`;
}

// CSS to style message alignment and timestamp positioning
const style = document.createElement("style");
style.textContent = `
    .message-wrapper {
        display: flex;
        flex-direction: column;
        max-width: 70%;
    }
    .sender-wrapper {
        align-items: flex-end;
        margin-left: auto;
    }
    .response-wrapper {
        align-items: flex-start;
    }
    .message-timestamp {
        font-size: 0.75rem;
        color: #666;
        margin-top: 4px;
    }
    .timestamp-right {
        margin-right: 15px;
        text-align: end;
    }
    .timestamp-left {
        margin-left: 20px;
        text-align: start;
    }
    .message.sender {
        border-radius: 10px;
        padding: 8px 12px;
        margin-bottom: 2px;
    }
`;
document.head.appendChild(style);

// Ensure input is focused when the page loads for the first time
// This should be at the end of the file to avoid conflicts

document.addEventListener("DOMContentLoaded", function () {
  const messageInput = document.getElementById("message-input");
  if (messageInput) messageInput.focus();
});
