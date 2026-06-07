/**
 * VERIFRAME — Video Authenticity Detector
 * jQuery frontend logic
 */

$(function () {

  /* ---- State ---- */
  let selectedFile = null;

  /* ---- Element refs ---- */
  const $dropZone    = $('#dropZone');
  const $videoInput  = $('#videoInput');
  const $filePreview = $('#filePreview');
  const $fileName    = $('#fileName');
  const $fileSize    = $('#fileSize');
  const $btnClear    = $('#btnClear');
  const $btnAnalyze  = $('#btnAnalyze');
  const $statusDot   = $('#statusDot');
  const $uploadPanel = $('#uploadPanel');
  const $progressPanel = $('#progressPanel');
  const $resultPanel = $('#resultPanel');
  const $errorToast  = $('#errorToast');

  /* ================================================
     FILE SELECTION
  ================================================ */
  $dropZone.on('click', function () { $videoInput[0].click(); });

  $videoInput.on('change', function () {
    if (this.files && this.files[0]) setFile(this.files[0]);
  });

  // Drag & drop
  $dropZone.on('dragover dragenter', function (e) {
    e.preventDefault();
    $(this).addClass('drag-over');
  });

  $dropZone.on('dragleave drop', function (e) {
    e.preventDefault();
    $(this).removeClass('drag-over');
  });

  $dropZone.on('drop', function (e) {
    const file = e.originalEvent.dataTransfer.files[0];
    if (file) setFile(file);
  });

  function setFile(file) {
    const allowed = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska',
                     'video/webm', 'video/x-flv', 'video/x-msvideo'];
    // Also check extension as browsers may not report correct MIME
    const ext = file.name.split('.').pop().toLowerCase();
    const allowedExt = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv'];

    if (!allowedExt.includes(ext)) {
      showError('Unsupported file type. Please upload: MP4, AVI, MOV, MKV, WEBM, or FLV');
      return;
    }

    if (file.size > 500 * 1024 * 1024) {
      showError('File too large. Maximum size is 500 MB.');
      return;
    }

    selectedFile = file;
    $fileName.text(file.name);
    $fileSize.text(formatBytes(file.size));
    $filePreview.show();
    $btnAnalyze.prop('disabled', false);
    $statusDot.addClass('ready').removeClass('active');
  }

  $btnClear.on('click', function (e) {
    e.stopPropagation();
    resetUpload();
  });

  function resetUpload() {
    selectedFile = null;
    $videoInput.val('');
    $filePreview.hide();
    $btnAnalyze.prop('disabled', true);
    $statusDot.removeClass('ready active');
  }

  /* ================================================
     ANALYZE
  ================================================ */
  $btnAnalyze.on('click', function () {
    if (!selectedFile) return;
    startAnalysis();
  });

  function startAnalysis() {
    // Show progress panel
    $uploadPanel.hide();
    $progressPanel.show();
    $resultPanel.hide();

    // Animate progress steps
    const steps = $('#progressSteps .step');
    let currentStep = 0;

    const stepDurations = [500, 800, 1000, 1200, 900, 600];
    let elapsed = 0;

    function activateStep(i) {
      steps.eq(i).addClass('active');
      const pct = ((i + 1) / steps.length) * 100;
      $('#progressBar').css('width', pct + '%');
    }

    activateStep(0);

    let stepTimer;
    function nextStep() {
      if (currentStep < steps.length - 1) {
        steps.eq(currentStep).removeClass('active').addClass('done');
        currentStep++;
        activateStep(currentStep);
        elapsed += stepDurations[currentStep] || 800;
        stepTimer = setTimeout(nextStep, stepDurations[currentStep] || 800);
      }
    }

    // Progress animation runs independently from actual upload
    stepTimer = setTimeout(nextStep, stepDurations[0]);

    // Build FormData
    const formData = new FormData();
    formData.append('video', selectedFile);

    $('#progressId').text('JOB-' + Math.random().toString(36).substr(2, 8).toUpperCase());

    // AJAX request
    $.ajax({
      url: '/analyze',
      type: 'POST',
      data: formData,
      processData: false,
      contentType: false,
      success: function (data) {
        clearTimeout(stepTimer);
        // Mark all steps done
        steps.removeClass('active').addClass('done');
        $('#progressBar').css('width', '100%');

        setTimeout(function () {
          $progressPanel.hide();
          renderResults(data);
          $resultPanel.show();
        }, 600);
      },
      error: function (xhr) {
        clearTimeout(stepTimer);
        $progressPanel.hide();
        $uploadPanel.show();
        resetUpload();

        let msg = 'Analysis failed. Please try again.';
        if (xhr.responseJSON && xhr.responseJSON.error) msg = xhr.responseJSON.error;
        showError(msg);
      }
    });
  }

  /* ================================================
     RENDER RESULTS
  ================================================ */
  function renderResults(data) {
    const verdict   = data.verdict;        // AUTHENTIC | SUSPICIOUS | MANIPULATED
    const score     = data.authenticity_score;
    const confidence = data.confidence;
    const meta      = data.metadata;
    const checks    = data.checks;
    const summary   = data.summary;
    let audio_confidence = 0;
    let visual_confidence = 0;

    audio_confidence = (data.checks.audio.is_fake) ? data.checks.audio.prob_fake : data.checks.audio.prob_real;
    visual_confidence = (data.checks.visual.is_fake) ? data.checks.visual.prob_fake : data.checks.visual.prob_real;

    /* -- Verdict banner -- */
    const $banner = $('#verdictBanner');
    const icons = { AUTHENTIC: '✔', SUSPICIOUS: '⚠', MANIPULATED: '✕' };
    const cls   = verdict.toLowerCase();

    $banner.removeClass('authentic suspicious manipulated').addClass(cls);
    $('#verdictIcon').text(icons[verdict] || '?');
    $('#verdictValue').text(verdict);
    $('#verdictConfidence').text('Confidence: ' + confidence);

    // Score ring animation
    const circumference = 314; // 2π * 50
    const offset = circumference - (score / 100) * circumference;
    $('#scoreRing').css('stroke-dashoffset', circumference);
    setTimeout(function () {
      $('#scoreRing').css('stroke-dashoffset', offset);
    }, 100);

    final_result = ((Number(audio_confidence) + Number(visual_confidence)) / 2) * 100;
    // Animate score number
    // animateNumber('#scoreNum', 0, Math.round(score), 1200);
    $('#scoreNum').text((final_result.toFixed(2) + '%'));

    /* -- Metadata grid -- */
    const metaItems = [
      { label: 'RESOLUTION',  value: meta.resolution },
      { label: 'FPS',         value: meta.fps },
      { label: 'DURATION',    value: meta.duration_seconds + 's' },
      { label: 'FRAMES',      value: meta.frame_count },
      { label: 'CODEC',       value: meta.codec || 'N/A' },
      { label: 'FILE SIZE',   value: meta.file_size_mb + ' MB' },
    ];

    const $metaGrid = $('#metaGrid').empty();
    metaItems.forEach(function (item) {
      $metaGrid.append(
        '<div class="info-card">' +
          '<div class="info-card-label">' + item.label + '</div>' +
          '<div class="info-card-value">' + item.value + '</div>' +
        '</div>'
      );
    });

    /* -- Check cards -- */
    const checkLabels = {
      metadata:    'METADATA INTEGRITY',
      noise:       'NOISE ANALYSIS',
      compression: 'COMPRESSION ARTIFACTS',
      temporal:    'TEMPORAL CONSISTENCY',
      ela:         'ERROR LEVEL ANALYSIS',
      copy_move:   'COPY-MOVE DETECTION',
    };

    const $checksGrid = $('#checksGrid').empty();
    $.each(checks, function (key, val) {
      const label  = checkLabels[key] || key.toUpperCase();
      const status = val.status || 'warning';
      const s      = val.score || 0;
      let confidence = (val.is_fake) ? val.prob_fake : val.prob_real;
      let result_part = (val.status === 'fail') ? 'AI Generate' : 'Real';

      $checksGrid.append(
        '<div class="check-card ' + status + '">' +
          '<div class="check-name">' + label + ' (Confidence Score)</div>' +
          '<div class="check-bar-wrap"><div class="check-bar" data-score="' + confidence * 100 + '" style="width:0%"></div></div>' +
          '<div class="check-score">' + confidence +
            '<span class="check-status">' + result_part + '</span>' +
          '</div>' +
        '</div>'
      );
    });

    // Animate bars
    setTimeout(function () {
      $('.check-bar').each(function () {
        $(this).css('width', $(this).data('score') + '%');
      });
    }, 150);

    /* -- Issues list -- */
    const issues = summary.issues_found || [];
    const $issuesPanel = $('#issuesPanel');
    const $issuesList  = $('#issuesList').empty();
    const $issuesTitle = $('#issuesTitle');

    if (issues.length === 0 || (issues.length === 1 && issues[0].includes('No significant'))) {
      $issuesPanel.addClass('clean');
      $issuesTitle.text('✔ No anomalies detected');
      $issuesList.append('<li>All forensic checks passed without significant findings.</li>');
    } else {
      $issuesPanel.removeClass('clean');
      $issuesTitle.text('⚠ Anomalies detected (' + issues.length + ')');
      issues.forEach(function (issue) {
        $issuesList.append('<li>' + escapeHtml(issue) + '</li>');
      });
    }
  }

  /* ================================================
     RESET
  ================================================ */
  $('#btnReset').on('click', function () {
    $resultPanel.hide();
    $progressPanel.hide();
    resetUpload();
    $('#progressSteps .step').removeClass('active done');
    $('#progressBar').css('width', '0%');
    $uploadPanel.show();
  });

  /* ================================================
     HELPERS
  ================================================ */
  function showError(msg) {
    $('#errorMsg').text(msg);
    $errorToast.stop(true, true).show().delay(4000).fadeOut(400);
  }

  function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  }

  function animateNumber(selector, from, to, duration) {
    const $el = $(selector);
    const start = Date.now();
    function tick() {
      const elapsed = Date.now() - start;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      $el.text(Math.round(from + (to - from) * ease));
      if (progress < 1) requestAnimationFrame(tick);
    }
    tick();
  }

  function escapeHtml(str) {
    return $('<div>').text(str).html();
  }

});
