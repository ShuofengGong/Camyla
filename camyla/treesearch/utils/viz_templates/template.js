/**
 * AI Scientist-v2 Experiment Visualization — List View
 */

hljs.highlightAll();

// ============================================================================
// Global State
// ============================================================================

const treeStructData = "PLACEHOLDER_TREE_DATA";

let currentStage = null;
let currentProposal = null;
let currentTreeData = null;
let selectedNodeIndex = -1;
let availableStages = [];

// ============================================================================
// Node List Rendering
// ============================================================================

function getFirstMetricSummary(metrics) {
  if (!metrics || !metrics.metric_names || metrics.metric_names.length === 0) return null;
  const parts = [];
  for (const m of metrics.metric_names) {
    if (m.data && m.data.length > 0) {
      const val = m.data[0].final_value;
      if (val != null) {
        const arrow = m.lower_is_better ? '\u2193' : '\u2191';
        parts.push(m.metric_name + ' ' + arrow + ' ' + val.toFixed(4));
      }
    }
  }
  return parts.length > 0 ? parts.join('  ') : null;
}

function renderNodeList(treeData) {
  currentTreeData = treeData;
  selectedNodeIndex = -1;

  const container = document.getElementById('node-list-container');
  const header = document.getElementById('node-list-header');

  // Clear previous rows (keep header)
  const existing = container.querySelectorAll('.node-row');
  existing.forEach(el => el.remove());

  if (!treeData || !treeData.plan) {
    container.innerHTML = '<div class="node-list-header">Experiments</div>' +
      '<div class="empty-state"><div class="empty-state-icon">--</div><p>No experiment data available</p></div>';
    return;
  }

  const nodeCount = treeData.plan.length;
  header.textContent = 'Experiments (' + nodeCount + ' nodes)';

  // Build parent_indices from edges if not directly available
  let parentIndices = treeData.parent_indices;
  if (!parentIndices) {
    parentIndices = new Array(nodeCount).fill(-1);
    if (treeData.edges) {
      for (const [parentIdx, childIdx] of treeData.edges) {
        if (childIdx >= 0 && childIdx < nodeCount) {
          parentIndices[childIdx] = parentIdx;
        }
      }
    }
  }

  for (let i = 0; i < nodeCount; i++) {
    const isBest = treeData.is_best_node?.[i] || false;
    const isBuggy = !!(treeData.exc_type?.[i]);
    const plan = treeData.plan[i] || '';
    const modSummary = treeData.modification_summaries?.[i] || '';
    const parentIdx = parentIndices[i];
    const metricsSummary = getFirstMetricSummary(treeData.metrics?.[i]);

    const row = document.createElement('div');
    row.className = 'node-row' + (isBest ? ' is-best' : '') + (isBuggy ? ' is-buggy' : '');
    row.setAttribute('data-index', i);

    // Step number circle
    const stepEl = document.createElement('div');
    stepEl.className = 'node-step';
    stepEl.textContent = i;
    row.appendChild(stepEl);

    // Middle info section
    const infoEl = document.createElement('div');
    infoEl.className = 'node-info';

    const planSummaryEl = document.createElement('div');
    planSummaryEl.className = 'node-plan-summary';
    const displayText = modSummary || plan.split('\n')[0];
    planSummaryEl.textContent = displayText.substring(0, 120) || (isBuggy ? 'Error' : 'Node ' + i);
    planSummaryEl.title = displayText;
    infoEl.appendChild(planSummaryEl);

    const metaEl = document.createElement('div');
    metaEl.className = 'node-meta';

    if (parentIdx >= 0) {
      const parentTag = document.createElement('span');
      parentTag.className = 'node-parent-tag';
      parentTag.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
        '<path d="M12 19V5M5 12l7-7 7 7"/></svg> from #' + parentIdx;
      metaEl.appendChild(parentTag);
    } else {
      const rootTag = document.createElement('span');
      rootTag.className = 'node-parent-tag';
      rootTag.textContent = 'root';
      metaEl.appendChild(rootTag);
    }

    if (isBest) {
      const bestBadge = document.createElement('span');
      bestBadge.className = 'status-badge success';
      bestBadge.textContent = 'Best';
      metaEl.appendChild(bestBadge);
    }
    if (isBuggy) {
      const bugBadge = document.createElement('span');
      bugBadge.className = 'status-badge error';
      bugBadge.textContent = treeData.exc_type[i] || 'Error';
      metaEl.appendChild(bugBadge);
    }

    infoEl.appendChild(metaEl);
    row.appendChild(infoEl);

    // Metric chip on the right
    if (metricsSummary) {
      const chipEl = document.createElement('div');
      chipEl.className = 'node-metric-chip';
      chipEl.innerHTML = metricsSummary.replace(
        /[\d.]+/g, m => '<span class="metric-value">' + m + '</span>'
      );
      row.appendChild(chipEl);
    }

    row.addEventListener('click', function () {
      selectNode(i);
    });

    container.appendChild(row);
  }

  // Auto-select best node, or first node
  const bestIdx = treeData.is_best_node ? treeData.is_best_node.indexOf(true) : -1;
  selectNode(bestIdx >= 0 ? bestIdx : 0);
}

function selectNode(index) {
  if (!currentTreeData) return;
  selectedNodeIndex = index;

  // Update row selection styles
  document.querySelectorAll('.node-row').forEach(row => {
    row.classList.remove('selected');
    if (parseInt(row.getAttribute('data-index')) === index) {
      row.classList.add('selected');
      row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  });

  updateNodeInfo(index, currentTreeData);
}

// ============================================================================
// UI Update Functions (right panel)
// ============================================================================

function updateNodeInfo(nodeIndex, treeData) {
  if (!treeData) return;

  const code = treeData.code?.[nodeIndex];
  const plan = treeData.plan?.[nodeIndex];
  const plotCode = treeData.plot_code?.[nodeIndex];
  const plotPlan = treeData.plot_plan?.[nodeIndex];
  const metrics = treeData.metrics?.[nodeIndex];
  const excType = treeData.exc_type?.[nodeIndex] || '';

  const rawExcInfo = treeData.exc_info?.[nodeIndex];
  let excInfo = '';
  if (rawExcInfo) {
    if (rawExcInfo.args && rawExcInfo.args[0]) {
      excInfo = rawExcInfo.args[0];
    } else if (rawExcInfo.error) {
      excInfo = rawExcInfo.error;
    } else if (rawExcInfo.stderr) {
      const lines = rawExcInfo.stderr.trim().split('\n');
      for (let i = lines.length - 1; i >= 0; i--) {
        if (/^\w+(Error|Exception):/.test(lines[i])) {
          excInfo = lines[i];
          break;
        }
      }
      if (!excInfo) excInfo = lines[lines.length - 1] || '';
    }
  }

  const rawExcStack = treeData.exc_stack?.[nodeIndex];
  let excStack = [];
  if (typeof rawExcStack === 'string') {
    excStack = rawExcStack.split('\n');
  } else if (Array.isArray(rawExcStack)) {
    excStack = rawExcStack;
  }
  const plots = treeData.plots?.[nodeIndex] || [];
  const datasets = treeData.datasets_successfully_tested?.[nodeIndex] || [];
  const execTimeFeedback = treeData.exec_time_feedback?.[nodeIndex] || '';
  const execTime = treeData.exec_time?.[nodeIndex] || '';

  // Build parent info string
  let parentInfo = '';
  const parentIndices = treeData.parent_indices;
  if (parentIndices && parentIndices[nodeIndex] >= 0) {
    parentInfo = ' (from #' + parentIndices[nodeIndex] + ')';
  } else if (treeData.edges) {
    for (const [p, c] of treeData.edges) {
      if (c === nodeIndex) { parentInfo = ' (from #' + p + ')'; break; }
    }
  }

  const isBest = treeData.is_best_node?.[nodeIndex];
  const titleEl = document.getElementById('info-title');
  if (titleEl) {
    titleEl.innerHTML = '<span>Node ' + nodeIndex + parentInfo + '</span>' +
      (isBest ? '<span class="node-badge">Best Result</span>' : '');
  }

  // Plan
  const planCard = document.getElementById('plan-card');
  const planEl = document.getElementById('plan');
  if (plan) {
    planCard.style.display = 'block';
    planEl.textContent = plan;
  } else {
    planCard.style.display = 'none';
  }

  // Execution Time
  const execTimeCard = document.getElementById('exec-time-card');
  const execTimeEl = document.getElementById('exec_time');
  const execTimeFeedbackEl = document.getElementById('exec_time_feedback');
  if (execTime) {
    execTimeCard.style.display = 'block';
    execTimeEl.textContent = execTime;
    execTimeFeedbackEl.textContent = execTimeFeedback || '';
  } else {
    execTimeCard.style.display = 'none';
  }

  // Exception
  const exceptionCard = document.getElementById('exception-card');
  if (excType) {
    exceptionCard.style.display = 'block';
    document.getElementById('exc_type_badge').textContent = excType;
    document.getElementById('exc_info').textContent = excInfo;
    const stderrContent = rawExcInfo?.stderr || '';
    const stackContent = excStack.length > 0
      ? excStack.map(item => {
          if (Array.isArray(item)) {
            return '  File "' + item[0] + '", line ' + item[1] + ', in ' + item[2] + '\n    ' + (item[3] || '');
          }
          return item;
        }).join('\n')
      : stderrContent;
    document.getElementById('exc_stack').textContent = stackContent;
  } else {
    exceptionCard.style.display = 'none';
  }

  // Metrics
  const metricsCard = document.getElementById('metrics-card');
  const metricsEl = document.getElementById('metrics');
  if (metrics && metrics.metric_names) {
    metricsCard.style.display = 'block';
    let metricsContent = '';
    for (const metric of metrics.metric_names) {
      metricsContent += '<div class="metric-group">' +
        '<div class="metric-name">' + metric.metric_name + '</div>' +
        '<div class="metric-description">' + (metric.description || 'No description') +
        ' (' + (metric.lower_is_better ? 'Lower is better' : 'Higher is better') + ')</div>' +
        '<table class="metric-table">' +
        '<tr><th>Dataset</th><th>Final Value</th><th>Best Value</th></tr>';
      for (const dataPoint of metric.data || []) {
        metricsContent += '<tr>' +
          '<td>' + dataPoint.dataset_name + '</td>' +
          '<td>' + (dataPoint.final_value?.toFixed(4) || 'N/A') + '</td>' +
          '<td>' + (dataPoint.best_value?.toFixed(4) || 'N/A') + '</td>' +
          '</tr>';
      }
      metricsContent += '</table></div>';
    }
    metricsEl.innerHTML = metricsContent;
  } else {
    metricsCard.style.display = 'none';
  }

  // Datasets Tested
  const datasetsCard = document.getElementById('datasets-card');
  const datasetsEl = document.getElementById('datasets_successfully_tested');
  if (datasets && datasets.length > 0) {
    datasetsCard.style.display = 'block';
    datasetsEl.innerHTML = datasets.map(d => '<span class="dataset-chip">' + d + '</span>').join('');
  } else {
    datasetsCard.style.display = 'none';
  }

  // Plots
  const plotsCard = document.getElementById('plots-card');
  const plotsEl = document.getElementById('plots');
  if (plots && plots.length > 0) {
    plotsCard.style.display = 'block';
    plotsEl.innerHTML = plots.map(p => '<div class="plot-item"><img src="' + p + '" alt="Plot" onerror="this.parentElement.style.display=\'none\'"/></div>').join('');
  } else {
    plotsCard.style.display = 'none';
  }

  // Code
  const codeCard = document.getElementById('code-card');
  const codeEl = document.getElementById('code');
  if (code) {
    codeCard.style.display = 'block';
    codeEl.innerHTML = hljs.highlight(code, { language: 'python' }).value;
  } else {
    codeCard.style.display = 'none';
  }

  // Plot Code
  const plotCodeCard = document.getElementById('plot-code-card');
  const plotCodeEl = document.getElementById('plot_code');
  if (plotCode) {
    plotCodeCard.style.display = 'block';
    plotCodeEl.innerHTML = hljs.highlight(plotCode, { language: 'python' }).value;
  } else {
    plotCodeCard.style.display = 'none';
  }

  // Plot Plan
  const plotPlanCard = document.getElementById('plot-plan-card');
  const plotPlanEl = document.getElementById('plot_plan');
  if (plotPlan) {
    plotPlanCard.style.display = 'block';
    plotPlanEl.textContent = plotPlan;
  } else {
    plotPlanCard.style.display = 'none';
  }
}

// ============================================================================
// Stage and Proposal Selection
// ============================================================================

function selectStage(stageId) {
  if (!availableStages.includes(stageId)) return;

  currentStage = stageId;

  document.querySelectorAll('.stage-tab').forEach(tab => {
    tab.classList.remove('active');
    if (tab.getAttribute('data-stage') === stageId) {
      tab.classList.add('active');
    }
  });

  const proposalSelector = document.getElementById('proposal-selector');
  if (stageId === 'Stage_2' && treeStructData.stage2_proposals && treeStructData.stage2_proposals.length > 0) {
    proposalSelector.classList.add('visible');

    if (!currentProposal) {
      const dropdown = document.getElementById('proposal-dropdown');
      if (dropdown.options.length > 0) {
        currentProposal = dropdown.value;
      }
    }

    loadProposalData(currentProposal);
  } else {
    proposalSelector.classList.remove('visible');

    if (treeStructData.stages && treeStructData.stages[stageId]) {
      renderNodeList(treeStructData.stages[stageId]);
    }
  }
}

function selectProposal(proposalDirName) {
  currentProposal = proposalDirName;
  loadProposalData(proposalDirName);
}

function loadProposalData(proposalDirName) {
  if (!proposalDirName) return;

  const proposalData = treeStructData.stage2_data?.[proposalDirName];
  if (proposalData) {
    renderNodeList(proposalData);
  } else {
    console.warn('No data found for proposal: ' + proposalDirName);
    const container = document.getElementById('node-list-container');
    container.innerHTML = '<div class="node-list-header">Experiments</div>' +
      '<div class="empty-state"><div class="empty-state-icon">--</div><p>No data for this proposal</p></div>';
  }
}

// ============================================================================
// Initialization
// ============================================================================

function initializeVisualization() {
  // Check if this is a single-stage (non-unified) data structure
  const isUnified = treeStructData.stages !== undefined;

  if (!isUnified) {
    // Single stage data — render directly
    availableStages = [];
    document.getElementById('stage-tabs').style.display = 'none';
    document.getElementById('proposal-selector').style.display = 'none';
    renderNodeList(treeStructData);
    return;
  }

  availableStages = treeStructData.completed_stages || [];

  document.querySelectorAll('.stage-tab').forEach(tab => {
    const stageId = tab.getAttribute('data-stage');
    if (availableStages.includes(stageId)) {
      tab.classList.remove('disabled');
    } else {
      tab.classList.add('disabled');
    }
  });

  // Populate proposal dropdown for Stage 2
  const dropdown = document.getElementById('proposal-dropdown');
  if (treeStructData.stage2_proposals && treeStructData.stage2_proposals.length > 0) {
    dropdown.innerHTML = '';
    for (const proposal of treeStructData.stage2_proposals) {
      const option = document.createElement('option');
      option.value = proposal.dir_name;

      let displayText = proposal.display_name;
      if (proposal.status === 'completed' && proposal.best_dice !== null) {
        displayText += ' (Dice: ' + proposal.best_dice.toFixed(4) + ')';
      } else if (proposal.status === 'in_progress') {
        displayText += ' (In Progress)';
      } else if (proposal.status === 'not_started') {
        displayText += ' (Not Started)';
      }

      option.textContent = displayText;
      option.disabled = !proposal.has_data;
      dropdown.appendChild(option);
    }
  }

  if (availableStages.length > 0) {
    selectStage(availableStages[0]);
  } else {
    const container = document.getElementById('node-list-container');
    container.innerHTML = '<div class="node-list-header">Experiments</div>' +
      '<div class="empty-state"><div class="empty-state-icon">--</div><p>No experiment data available</p></div>';
  }
}

document.addEventListener('DOMContentLoaded', initializeVisualization);

if (document.readyState === 'complete' || document.readyState === 'interactive') {
  setTimeout(initializeVisualization, 0);
}
