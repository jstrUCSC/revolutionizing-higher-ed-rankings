let data = [];
let facultyData = [];
let categories = [];
let currentPage = 1;
const rowsPerPage = 25;
let selectedRegion = 'all';
let lastFiltered = [];
let selectedCountry = 'all';
let expandedRows = new Set();

const ACTIVE_FIELDS = [
    "Machine Learning",
    "Computer Vision & Image Processing",
    "Natural Language Processing",
];

const DISPLAY_LABELS = {
    'Machine Learning': 'Machine Learning',
    'Computer Vision & Image Processing': 'Computer Vision & Image Processing',
    'Natural Language Processing': 'Natural Language Processing',
};

const EPS = 1e-9;
const NORMALIZE = 'unit-variance';
let fieldStats = {};

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
    data = await loadCSV('3_f_1.csv');
    facultyData = await loadCSV('3_faculty_score.csv');

    // Extract categories dynamically
    if (data.length > 0) {
        const columns = Object.keys(data[0]);
        categories = columns.slice(2); // Assume categories start at index 2
    }

    computeFieldStats();
    displayFilters();
    setupRegionFilter();
    setupCountryFilter();
    displayRankings();
}

function setupCountryFilter() {
    const sel = document.getElementById('countryFilter');
    const countries = Array.from(new Set(
        data.map(r => (r.Country || '').trim()).filter(s => s && s.toLowerCase() !== 'unknown')
    )).sort((a, b) => a.localeCompare(b));

    sel.innerHTML = '<option value="all">All Countries</option>' +
        countries.map(c => `<option value="${c}">${c}</option>`).join('');

    sel.addEventListener('change', () => {
        selectedCountry = sel.value;
        resetPageAndDisplayRankings();
    });
}

function computeFieldStats() {
    fieldStats = {};
    const cols = ACTIVE_FIELDS;
    cols.forEach(cat => {
        const vals = data.map(r => parseFloat(r[cat])).filter(v => !isNaN(v) && v > 0);
        const n = vals.length;
        const mean = n ? vals.reduce((a,b)=>a+b,0)/n : 0;
        const varSample = n>1 ? vals.reduce((a,b)=>a+(b-mean)*(b-mean),0)/(n-1) : 0;
        const std = Math.sqrt(varSample);
        fieldStats[cat] = { mean, std };
    });
}

function displayFilters() {
    const table = document.getElementById('filterTable');
    const tableBody = table.querySelector('tbody');
    tableBody.innerHTML = '';

    // Add active fields with checkboxes
    ACTIVE_FIELDS.forEach(category => {
        const label = DISPLAY_LABELS[category] || category;
        const safeId = toId(category);
        const row = tableBody.insertRow();
        row.innerHTML = `
            <td>${label}</td>
            <td>
                <label class="switch">
                    <input
                        id="${safeId}"
                        class="field-checkbox"
                        type="checkbox"
                        data-field="${category}"
                        checked
                        onclick="resetPageAndDisplayRankings()"
                    >
                    <span class="slider round"></span>
                </label>
            </td>`;
    });

    // Add coming soon row
    const comingRow = tableBody.insertRow();
    comingRow.innerHTML = `
        <td>Coming Soon</td>
        <td>
            <label class="switch">
                <input type="checkbox" disabled>
                <span class="slider round"></span>
            </label>
        </td>`;

    updateToggleAllButtonLabel();
}

function toId(name) {
    return 'field-' + name.toLowerCase().replace(/[^a-z0-9]+/g, '-');
}

function updateToggleAllButtonLabel() {
    const boxes = document.querySelectorAll('.field-checkbox');
    const allChecked = Array.from(boxes).every(b => b.checked);
    const btn = document.getElementById('toggleAll');
    if (btn) btn.textContent = allChecked ? 'Deselect All' : 'Select All';
}

function setupRegionFilter() {
    const regionFilter = document.getElementById('regionFilter');
    regionFilter.addEventListener('change', resetPageAndDisplayRankings);
}

function resetPageAndDisplayRankings() {
    currentPage = 1;
    selectedRegion = document.getElementById('regionFilter').value;
    updateToggleAllButtonLabel();
    expandedRows.clear(); // Clear expanded rows when filters change
    displayRankings();
}

function getSelectedCategories() {
    return [...document.querySelectorAll('.field-checkbox:checked')]
           .map(el => el.dataset.field);
}

function getRawScore(univ, field) {
    const row = data.find(e => e.University === univ);
    if (!row) return 0;
    const s = parseFloat(row[field]);
    return isNaN(s) ? 0 : s;
}

function getNormalizedScore(univ, field) {
    const s = getRawScore(univ, field);
    if (!(s > 0)) return 0;
    const stats = fieldStats[field] || { std: 1 };
    const std = stats.std > EPS ? stats.std : 1;
    return s / std;
}

function getFacultyForUniversity(universityName, selectedCategories) {
    // Filter faculty data for this university and selected categories
    const filteredFaculty = facultyData.filter(faculty => {
        const matchesUniversity = faculty.University === universityName;
        const matchesCategory = selectedCategories.length === 0 || 
                                selectedCategories.includes(faculty.Category);
        return matchesUniversity && matchesCategory;
    });

    // Group by faculty name and sum scores
    const facultyMap = new Map();
    
    filteredFaculty.forEach(faculty => {
        const name = faculty['Faculty Name'] || 'Unknown';
        const score = parseFloat(faculty.Score) || 0;
        const category = faculty.Category || 'Unknown';
        
        if (!facultyMap.has(name)) {
            facultyMap.set(name, {
                name: name,
                categories: [],
                totalScore: 0
            });
        }
        
        const facultyInfo = facultyMap.get(name);
        facultyInfo.totalScore += score;
        if (!facultyInfo.categories.includes(category)) {
            facultyInfo.categories.push(category);
        }
    });
    
    // Convert to array and sort by total score
    const mergedFaculty = Array.from(facultyMap.values())
        .sort((a, b) => b.totalScore - a.totalScore);
    
    return mergedFaculty;
}

function toggleUniversityDropdown(universityName, rowElement) {
    const isExpanded = expandedRows.has(universityName);
    
    if (isExpanded) {
        // Collapse: Remove the details row
        const detailsRow = rowElement.nextElementSibling;
        if (detailsRow && detailsRow.classList.contains('faculty-details-row')) {
            detailsRow.remove();
        }
        expandedRows.delete(universityName);
        rowElement.querySelector('.expand-icon').innerHTML = '▶';
    } else {
        // Expand: Add the details row
        const selectedCategories = getSelectedCategories();
        const facultyList = getFacultyForUniversity(universityName, selectedCategories);
        
        const detailsRow = document.createElement('tr');
        detailsRow.classList.add('faculty-details-row');
        
        let facultyHTML = '<td colspan="3"><div class="faculty-details">';
        
        if (facultyList.length > 0) {
            facultyHTML += '<table class="faculty-table">';
            
            // Adjust header based on number of selected categories
            const fieldHeader = selectedCategories.length === 1 ? 'Field' : 'Fields';
            facultyHTML += `<thead><tr><th>Faculty Name</th><th>${fieldHeader}</th><th>Total Contribution</th></tr></thead>`;
            facultyHTML += '<tbody>';
            
            facultyList.forEach(faculty => {
                const fieldsDisplay = faculty.categories.join(', ');
                facultyHTML += `
                    <tr>
                        <td>${faculty.name}</td>
                        <td>${fieldsDisplay}</td>
                        <td>${faculty.totalScore.toFixed(4)}</td>
                    </tr>
                `;
            });
            
            facultyHTML += '</tbody></table>';
        } else {
            facultyHTML += '<p>No faculty data available for the selected fields.</p>';
        }
        
        facultyHTML += '</div></td>';
        detailsRow.innerHTML = facultyHTML;
        
        rowElement.insertAdjacentElement('afterend', detailsRow);
        expandedRows.add(universityName);
        rowElement.querySelector('.expand-icon').innerHTML = '▼';
    }
}

function displayRankings() {
    const calculatedScores = [];
    const seenUniversities = new Set();

    data.forEach(university => {
        const selectedCats = getSelectedCategories();

        let totalScore = 0;
        if (selectedCats.length >= 2) {
            // Normalized harmonic mean for multiple fields
            let denom = 0, count = 0;
            selectedCats.forEach(cat => {
                const x = getNormalizedScore(university.University, cat);
                if (x > 0) { 
                    denom += 1 / (x + EPS); 
                    count++;
                }
            });
            totalScore = count > 0 ? (count / denom) : 0;
        } else if (selectedCats.length === 1) {
            // Raw score for single field
            totalScore = getRawScore(university.University, selectedCats[0]);
        } else {
            totalScore = 0;
        }

        // Apply region filtering
        if (selectedRegion !== 'all' && university.Continent) {
            const continentMatch = university.Continent.trim().toLowerCase() === selectedRegion.toLowerCase();
            if (!continentMatch) return;
        }
        
        // Apply countries filtering
        if (selectedCountry !== 'all') {
            const country = (university.Country || '').trim();
            if (!country || country !== selectedCountry) return;
        }

        if (!seenUniversities.has(university.University) && totalScore > 0) {
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
    tableBody.innerHTML = '';

    const start = (page - 1) * rowsPerPage;
    const end = page * rowsPerPage;

    data.slice(start, end).forEach((university, index) => {
        const row = tableBody.insertRow();
        row.classList.add('university-row');
        
        const universityCell = `
            <td>${start + index + 1}</td>
            <td class="university-name-cell">
                <span class="expand-icon">▶</span>
                <span class="university-name" onclick="toggleUniversityDropdown('${university.University.replace(/'/g, "\\'")}', this.closest('tr'))">${university.University}</span>
            </td>
            <td>${university.Score.toFixed(2)}</td>
        `;
        
        row.innerHTML = universityCell;
        
        // Re-expand if this university was previously expanded
        if (expandedRows.has(university.University)) {
            setTimeout(() => {
                toggleUniversityDropdown(university.University, row);
            }, 0);
        }
    });
}

function updatePageIndicator(page, totalPages) {
    document.getElementById('pageIndicator').textContent = `Page ${page} of ${totalPages}`;
}

function prevPage() {
    if (currentPage > 1) {
        currentPage--;
        displayPage(currentPage, lastFiltered);
        updatePageIndicator(currentPage, Math.ceil(lastFiltered.length / rowsPerPage));
    }
}

function nextPage() {
    const totalPages = Math.ceil(lastFiltered.length / rowsPerPage);
    if (currentPage < totalPages) {
        currentPage++;
        displayPage(currentPage, lastFiltered);
        updatePageIndicator(currentPage, totalPages);
    }
}

function toggleAllCheckboxes() {
    const boxes = document.querySelectorAll('.field-checkbox');
    const allChecked = [...boxes].every(b => b.checked);
    boxes.forEach(b => b.checked = !allChecked);
    updateToggleAllButtonLabel();
    resetPageAndDisplayRankings();
}

// Initialize the application
initialize();