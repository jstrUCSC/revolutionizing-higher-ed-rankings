let data = [];
let categories = [];
let currentPage = 1;
const rowsPerPage = 25; // Set the number of rows per page
let selectedRegion = 'all'; // Default to 'all' regions
let lastFiltered = []; // 页面总分数

const ACTIVE_FIELDS = [
    "Machine Learning",
    "Computer Vision & Image Processing",
    ];

const EPS = 1e-9;                  // 防止除零
const NORMALIZE = 'unit-variance'; // 归一化策略：'unit-variance'（默认）或 'mean'
let fieldStats = {};               // 存每个领域的 mean、std


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

async function initialize() {
    // data = await loadCSV('university_rankings.csv');
    data = await loadCSV('2_f.csv');

    // Extract categories dynamically
    if (data.length > 0) {
        const columns = Object.keys(data[0]);
        categories = columns.slice(2); // Assume categories start at index 2
    }

    computeFieldStats();

    displayFilters();       // Fields
    setupRegionFilter();    // Set up the region filter
    displayRankings();
}

function computeFieldStats() {
    const cols = categories.filter(c => c !== 'Continent');
    fieldStats = {};
    cols.forEach(cat => {
      const vals = data.map(r => parseFloat(r[cat]) || 0);
      const n = vals.length;
      const mean = vals.reduce((a, b) => a + b, 0) / Math.max(n, 1);
      // 样本方差（n-1）
      const varSample = vals.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / Math.max(n - 1, 1);
      const std = Math.sqrt(varSample);
      fieldStats[cat] = { mean, std };
    });
  }
  

function displayFilters() {
    const table = document.getElementById('filterTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = ''; // Clear existing rows

    // Filter categories to exclude "Continent" and any non-research fields
    const filteredCategories = categories.filter(category => category !== 'Continent');
    
    filteredCategories.forEach(category => {
        // const row = tableBody.insertRow();
        const isActive = ACTIVE_FIELDS.includes(category);

        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>
                ${category}${isActive ? '': '<span class="coming">(coming soon)</span>'}
            </td>
            <td>
                <label class="switch">
                    <input id="${category}" type="checkbox" ${isActive ? 'checked' : 'disabled'} onclick="resetPageAndDisplayRankings()">
                    <span class="slider round"></span>
                </label>
            </td>`;
    });

    updateToggleAllButtonLabel();

}

function updateToggleAllButtonLabel() {
    const boxes = document.querySelectorAll('#filterTable input[type="checkbox"]:not([disabled])');
    const allChecked = Array.from(boxes).every(b => b.checked);
    const btn = document.getElementById('toggleAll');
    if (btn) btn.textContent = allChecked ? 'Deselect All' : 'Select All';
}

function setupRegionFilter() {
    const regionFilter = document.getElementById('regionFilter');
    regionFilter.addEventListener('change', resetPageAndDisplayRankings); // Add listener for region change
}

function resetPageAndDisplayRankings() {
    currentPage = 1;
    selectedRegion = document.getElementById('regionFilter').value; // Get selected region
    updateToggleAllButtonLabel();
    displayRankings();
}

function displayRankings() {
    const calculatedScores = [];
    const seenUniversities = new Set(); // Set to track universities that have been added

    data.forEach(university => {
        // const totalScore = categories.reduce((sum, category) => {
        //     const checkbox = document.getElementById(category);
        //     if (checkbox?.checked) {
        //         sum += getScore(university.University, category);
        //     }
        //     return sum;
        // }, 0);
        const checkedCats = categories.filter(cat => {
            const el = document.getElementById(cat);
            return el && el.checked; // disabled 的不会被勾选到
        });
        
        let denom = 0; // 调和平均的分母项累加
        let count = 0; // 参与的领域个数
        checkedCats.forEach(cat => {
            const x = getNormalizedScore(university.University, cat); // 归一化后的 x_ij
            if (x > 0) {
                denom += 1 / (x + EPS);
                count++;
            }
        });
        const totalScore = count > 0 ? (count / denom) : 0;

        // Apply region filtering: Make sure to handle case where Continent is empty or invalid
        if (selectedRegion !== 'all' && university.Continent) {
            // Ensure case-insensitive comparison and handle missing regions
            const continentMatch = university.Continent.trim().toLowerCase() === selectedRegion.toLowerCase();
            if (!continentMatch) {
                return; // Skip this university if it doesn't match the selected region
            }
        }

        if (!seenUniversities.has(university.University)) {
            calculatedScores.push({
                University: university.University,
                Continent: university.Continent || 'Unknown',
                Score: totalScore
            });
            seenUniversities.add(university.University);
        }
    });

    calculatedScores.sort((a, b) => b.Score - a.Score);

    lastFiltered = calculatedScores;
    displayPage(currentPage, calculatedScores);
    updatePageIndicator(currentPage, Math.ceil(calculatedScores.length / rowsPerPage));
}

function displayPage(page, data) {
    const table = document.getElementById('rankingTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = ''; // Clear existing rows

    const start = (page - 1) * rowsPerPage;
    const end = page * rowsPerPage;

    data.slice(start, end).forEach((university, index) => {
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>${start + index + 1}</td>
            <td>${university.University}</td>
            <td>${university.Score.toFixed(2)}</td>`;
    });
}

function updatePageIndicator(page, totalPages) {
    document.getElementById('pageIndicator').textContent = `Page ${page} of ${totalPages}`;
}

function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        displayRankings();
    }
}

function nextPage() {
    const totalPages = Math.ceil((lastFiltered?.length || 0) / rowsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        displayRankings();
    }
}

// function getScore(universityName, categoryName) {
//     const row = data.find(entry => entry.University === universityName);
//     if (!row) return 0;

//     const score = parseFloat(row[categoryName]);
//     return isNaN(score) ? 0 : score;
// }
function getNormalizedScore(universityName, categoryName) {
    const row = data.find(entry => entry.University === universityName);
    if (!row) return 0;
  
    const score = parseFloat(row[categoryName]);
    if (!(score > 0)) return 0; 
  
    const stats = fieldStats[categoryName] || { mean: 0, std: 0 };
  
    if (NORMALIZE === 'mean') {
      const denom = stats.mean > EPS ? stats.mean : 1;
      return score / denom; // 仅除均值（不保证方差统一）
    } else { // 'unit-variance'（推荐）
      const denom = stats.std > EPS ? stats.std : 1;
      return score / denom; // 等方差
    }
  }

function toggleAllCheckboxes() {
    const checkboxes = document.querySelectorAll('#filterTable input[type="checkbox"]:not([disabled])');
    const allChecked = Array.from(checkboxes).every(checkbox => checkbox.checked);

    checkboxes.forEach(checkbox => checkbox.checked = !allChecked);
    // document.getElementById('toggleAll').textContent = allChecked ? 'Select All' : 'Deselect All';
    updateToggleAllButtonLabel();
    resetPageAndDisplayRankings(); // Re-display rankings after toggle
}

// Initialize the application at the end of the script
initialize();



