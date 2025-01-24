// Dynamically load CSV data from an external file
let data = [];
let categories = [];

/**
 * Loads a CSV file from a given URL and parses it into a list of objects.
 *
 * @param {string} filePath - The URL of the CSV file to load.
 *
 * @returns {Promise<Object[]>} - A promise that resolves with an array of objects, where each object
 *   is a row in the CSV file, with keys corresponding to the header names.
 */
async function loadCSV(filePath) {
    const response = await fetch(filePath);
    const csvText = await response.text();
    
    const rows = csvText.trim().split('\n');
    const headers = rows.shift().split(',');

    return rows.map(row => {
        const rowData = row.split(',');
        return headers.reduce((obj, header, index) => {
            obj[header.trim()] = rowData[index]?.trim() || '';
            return obj;
        }, {});
    });
}

/**
 * Initializes the script by loading the CSV data and extracting the categories.
 * The categories are used to generate the filter checkboxes in the UI.
 * @returns {Promise<void>}
 */
async function initialize() {
    data = await loadCSV('/nfs/stak/users/munam/hpc-share/revolutionizing-higher-ed-rankings/universities_ranked.csv');

    // Extract categories dynamically
    if (data.length > 0) {
        const columns = Object.keys(data[0]);
        categories = columns.slice(2); // Assume categories start at index 2
    }

    displayFilters();
}

/**
 * Generates the filter checkboxes in the UI based on the categories found in the CSV data.
 * The generated HTML is inserted into the <tbody> element of the #filterTable element.
 */
function displayFilters() {
    const table = document.getElementById('filterTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = ''; // Clear existing rows

    categories.forEach(category => {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>${category}</td>
            <td>
                <label class="switch">
                    <input id="${category}" type="checkbox" onclick="displayRankings()">
                    <span class="slider round"></span>
                </label>
            </td>`;
    });
}

/**
 * Toggles all the filter checkboxes at once. If all checkboxes are currently
 * checked, it will uncheck all of them. If not all checkboxes are checked, it
 * will check all of them. The text of the toggle button will be updated to
 * reflect the new state.
 */
function toggleAllCheckboxes() {
    const checkboxes = document.querySelectorAll('#filterTable input[type="checkbox"]');
    const allChecked = Array.from(checkboxes).every(checkbox => checkbox.checked);

    checkboxes.forEach(checkbox => checkbox.checked = !allChecked);
    document.getElementById('toggleAll').textContent = allChecked ? 'Select All' : 'Deselect All';

    displayRankings();
}

/**
 * Updates the rankings table in the UI based on the selected categories.
 * It calculates scores for each university by summing up the scores of
 * selected categories and displays the sorted results in the table.
 * The table is cleared and repopulated with the new ranking data.
 */

function displayRankings() {
    const calculatedScores = data.map(university => {
        const totalScore = categories.reduce((sum, category) => {
            const checkbox = document.getElementById(category);
            if (checkbox?.checked) {
                sum += getScore(university.University, category);
            }
            return sum;
        }, 0);

        return { University: university.University, Continent: university.Continent || 'Unknown', Score: totalScore };
    });

    calculatedScores.sort((a, b) => b.Score - a.Score);

    const table = document.getElementById('rankingTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = ''; // Clear existing rows

    calculatedScores.forEach((university, index) => {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${university.University}</td>
            <td>${university.Score.toFixed(2)}</td>`;
    });
}

/**
 * Returns the score of a university in a given category. If the university is
 * not found or the score is not a number, returns 0.
 *
 * @param {string} universityName - The name of the university to get the score for.
 * @param {string} categoryName - The name of the category to get the score from.
 * @returns {number} The score of the university in the category, or 0 if not found.
 */
function getScore(universityName, categoryName) {
    const row = data.find(entry => entry.University === universityName);
    if (!row) return 0;

    const score = parseFloat(row[categoryName]);
    return isNaN(score) ? 0 : score;
}

// Initialize the application
initialize();
