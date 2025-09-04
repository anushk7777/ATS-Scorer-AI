// Initialize PDF.js worker
        pdfjsLib.GlobalWorkerOptions.workerSrc = 'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
        
        // Google Gemini API Configuration
        // AI-powered resume analysis using Google Gemini Pro
        // Enhances ATS scoring with intelligent content analysis
        const GEMINI_API_CONFIG = {
            apiKey: 'YOUR_ACTUAL_GEMINI_API_KEY_HERE', // Replace this with your actual Gemini API key
            baseUrl: 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
            model: 'gemini-pro'
        };

        /**
         * Optimized scoring configuration for enhanced accuracy
         * Based on industry standards and ATS best practices
         */
        const SCORING_CONFIG = {
            // Main category weights (should sum to 1.0)
            categoryWeights: {
                format: 0.20,      // Reduced from 0.25 - basic requirement
                content: 0.35,     // Increased from 0.30 - most important for human reviewers
                keywords: 0.30,    // Increased from 0.25 - critical for ATS filtering
                readability: 0.15  // Reduced from 0.20 - important but not critical
            },
            
            // AI vs Rule-based scoring weights
            aiIntegration: {
                aiWeight: 0.65,        // Increased AI influence for better accuracy
                ruleBasedWeight: 0.35, // Reduced rule-based weight
                minAiScore: 30,        // Minimum AI score to trust AI analysis
                maxBonusPoints: 15     // Maximum bonus points from competitive/industry analysis
            },
            
            // Format analysis scoring breakdown
            formatScoring: {
                contactInfo: {
                    email: 20,
                    phone: 15,
                    linkedin: 10  // New: LinkedIn profile bonus
                },
                sections: {
                    experience: 25,
                    education: 20,
                    skills: 20
                },
                structure: {
                    clearHeadings: 10,  // New: Clear section headings
                    consistentFormatting: 10  // New: Consistent formatting
                }
            },
            
            // Content analysis scoring breakdown
            contentScoring: {
                wordCount: {
                    minimum: 200,
                    optimal: 400,
                    points: 25
                },
                actionVerbs: {
                    minimum: 3,
                    optimal: 6,
                    points: 25
                },
                quantifiableAchievements: {
                    minimum: 3,
                    optimal: 6,
                    points: 30
                },
                aiEnhancement: 20  // Points for AI-detected strengths
            },
            
            // Keyword analysis scoring breakdown
            keywordScoring: {
                coreKeywords: {
                    maxScore: 40,
                    threshold: 5
                },
                technicalKeywords: {
                    maxScore: 30,
                    threshold: 3
                },
                softSkills: {
                    maxScore: 20,
                    threshold: 4
                },
                aiIntegration: {
                    aiWeight: 0.70,
                    ruleWeight: 0.30
                },
                excellentThreshold: 85,
                improvementThreshold: 60
            },
            
            // Readability scoring breakdown
            readabilityScoring: {
                specialCharacters: {
                    maxScore: 30,
                    maxAllowed: 20,
                    penaltyPerChar: 2,
                    maxPenalty: 25
                },
                contentDensity: {
                    maxScore: 35,
                    minLineLength: 10,
                    optimalRatio: 0.3,
                    maxRatio: 0.5,
                    penaltyMultiplier: 50
                },
                atsCompatibility: {
                    maxScore: 35
                },
                aiIntegration: {
                    aiWeight: 0.70,
                    ruleWeight: 0.30,
                    readabilityWeight: 0.60,
                    atsWeight: 0.40
                },
                excellentThreshold: 85,
                improvementThreshold: 60
            },
            
            // Bonus scoring criteria
            bonusScoring: {
                competitivePosition: {
                    above_average: 5,
                    top_tier: 10
                },
                industryAlignment: {
                    threshold: 80,
                    bonus: 5
                },
                comprehensiveProfile: 3  // Complete profile with all sections
            },
            
            // Validation thresholds
            validationThresholds: {
                excellent: 85,
                good: 70,
                average: 55,
                needsImprovement: 40
            }
        };

        // Add property aliases for backward compatibility with existing function calls
        SCORING_CONFIG.contactInfo = SCORING_CONFIG.formatScoring.contactInfo;
        SCORING_CONFIG.sections = SCORING_CONFIG.formatScoring.sections;
        SCORING_CONFIG.structure = SCORING_CONFIG.formatScoring.structure;
        SCORING_CONFIG.wordCount = SCORING_CONFIG.contentScoring.wordCount;
        SCORING_CONFIG.actionVerbs = SCORING_CONFIG.contentScoring.actionVerbs;
        SCORING_CONFIG.quantifiableAchievements = SCORING_CONFIG.contentScoring.quantifiableAchievements;
        SCORING_CONFIG.industryAlignment = { threshold: 80 }; // Add missing industryAlignment config
        
        // Add missing keyword configuration properties
        SCORING_CONFIG.keywordScoring.coreKeywords.minRequired = SCORING_CONFIG.keywordScoring.coreKeywords.threshold;
        SCORING_CONFIG.keywordScoring.technicalKeywords.minRequired = SCORING_CONFIG.keywordScoring.technicalKeywords.threshold;
        SCORING_CONFIG.keywordScoring.softSkills.minRequired = SCORING_CONFIG.keywordScoring.softSkills.threshold;
        SCORING_CONFIG.keywordScoring.aiIntegration.weight = SCORING_CONFIG.keywordScoring.aiIntegration.aiWeight;
        SCORING_CONFIG.keywordScoring.aiIntegration.fallbackWeight = SCORING_CONFIG.keywordScoring.aiIntegration.ruleWeight;

        /**
         * Check if Gemini API is properly configured
         * @returns {boolean} True if API key is configured
         */
        function isGeminiConfigured() {
            return GEMINI_API_CONFIG.apiKey && 
                   GEMINI_API_CONFIG.apiKey !== 'YOUR_GEMINI_API_KEY_HERE' && 
                   GEMINI_API_CONFIG.apiKey !== 'YOUR_ACTUAL_GEMINI_API_KEY_HERE' && 
                   GEMINI_API_CONFIG.apiKey !== 'AIzaSyDXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX';
        }

        /**
         * Analyze resume content using Google Gemini AI
         * Provides intelligent insights for ATS optimization
         * @param {string} resumeText - The extracted resume text
         * @returns {Promise<Object|null>} AI analysis results or null if unavailable
         */
        async function analyzeWithGemini(resumeText) {
            if (!isGeminiConfigured()) {
                console.log('Gemini API not configured, using rule-based analysis only');
                updateProcessingStatus('AI analysis unavailable - proceeding with rule-based analysis');
                return null;
            }

            try {
                updateProcessingStatus('Running AI analysis with Google Gemini...');
                
                const prompt = `
                You are an expert ATS (Applicant Tracking System) analyzer and career consultant. Analyze this resume with the precision of a hiring manager and ATS system combined.
                
                Resume Text:
                ${resumeText}
                
                Provide a comprehensive JSON analysis with this exact structure:
                {
                    "strengths": ["specific resume strengths with examples"],
                    "improvements": ["actionable, specific improvements with clear instructions"],
                    "keywordOptimization": {
                        "score": number (0-100),
                        "feedback": "detailed analysis of keyword density, relevance, and industry alignment",
                        "missingKeywords": ["industry-specific keywords that should be added"],
                        "keywordDensity": "assessment of current keyword usage"
                    },
                    "readabilityScore": number (0-100),
                    "professionalSummary": "detailed feedback on summary/objective effectiveness and ATS optimization",
                    "missingSkills": ["high-demand skills commonly expected in this field"],
                    "industryAlignment": {
                        "detectedIndustry": "primary industry/field detected",
                        "alignmentScore": number (0-100),
                        "recommendations": "how to better align with industry standards"
                    },
                    "atsCompatibility": {
                        "formatScore": number (0-100),
                        "parseabilityIssues": ["specific formatting problems that ATS might struggle with"],
                        "sectionOrganization": "feedback on resume structure and section order"
                    },
                    "competitiveAnalysis": {
                        "marketPosition": "how this resume compares to market standards",
                        "differentiators": ["unique strengths that set candidate apart"],
                        "gaps": ["areas where candidate falls behind market expectations"]
                    },
                    "overallRecommendation": "comprehensive recommendation with priority actions"
                }
                
                Analysis Guidelines:
                1. Be specific and actionable in all feedback
                2. Consider current job market trends and ATS technology
                3. Identify the most likely industry/role based on content
                4. Provide keyword suggestions based on industry standards
                5. Flag any formatting that could cause ATS parsing issues
                6. Compare against current hiring manager expectations
                7. Prioritize improvements by impact on ATS success
                `;

                const response = await fetch(`${GEMINI_API_CONFIG.baseUrl}?key=${GEMINI_API_CONFIG.apiKey}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        contents: [{
                            parts: [{
                                text: prompt
                            }]
                        }]
                    })
                });

                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    console.error('Gemini API error:', response.status, errorData);
                    
                    // Provide specific error messages based on status code
                    if (response.status === 400) {
                        updateProcessingStatus('AI analysis failed: Invalid API key or request format');
                    } else if (response.status === 429) {
                        updateProcessingStatus('AI analysis failed: API quota exceeded');
                    } else if (response.status >= 500) {
                        updateProcessingStatus('AI analysis failed: Service temporarily unavailable');
                    } else {
                        updateProcessingStatus('AI analysis failed: Network error');
                    }
                    
                    return null;
                }

                const data = await response.json();
                const aiResponse = data.candidates?.[0]?.content?.parts?.[0]?.text;
                
                if (!aiResponse) {
                    console.error('Invalid response format from Gemini API');
                    return null;
                }

                // Parse JSON response from AI
                try {
                    const jsonMatch = aiResponse.match(/\{[\s\S]*\}/);
                    if (jsonMatch) {
                        return JSON.parse(jsonMatch[0]);
                    }
                } catch (parseError) {
                    console.error('Failed to parse AI response as JSON:', parseError);
                }
                
                return null;
            } catch (error) {
                console.error('Error calling Gemini API:', error);
                updateProcessingStatus('AI analysis failed - proceeding with rule-based analysis');
                return null;
            }
        }

        // Global variables
        let currentFile = null;
        let analysisResults = null;

        // DOM elements
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');
        const fileInfo = document.getElementById('file-info');
        const fileName = document.getElementById('file-name');
        const fileSize = document.getElementById('file-size');
        const removeFileBtn = document.getElementById('remove-file');
        const analyzeBtn = document.getElementById('analyze-btn');
        const uploadSection = document.getElementById('upload-section');
        const processingSection = document.getElementById('processing-section');
        const resultsSection = document.getElementById('results-section');
        const processingStatus = document.getElementById('processing-status');
        const progressBar = document.getElementById('progress-bar');

        // Dynamic status messages for enhanced user experience
        const DYNAMIC_STATUS_MESSAGES = {
            initializing: [
                'Initializing AI analysis engine...',
                'Preparing resume scanner...',
                'Loading ATS optimization algorithms...'
            ],
            aiAnalysis: [
                'Running AI analysis with Google Gemini...',
                'Processing content with advanced AI...',
                'Analyzing resume with machine learning...',
                'Extracting insights with neural networks...'
            ],
            formatAnalysis: [
                'Analyzing format and structure...',
                'Checking ATS compatibility...',
                'Evaluating document formatting...',
                'Scanning for parsing issues...'
            ],
            contentAnalysis: [
                'Evaluating content quality...',
                'Assessing professional experience...',
                'Analyzing skill descriptions...',
                'Reviewing achievement statements...'
            ],
            keywordAnalysis: [
                'Checking keyword optimization...',
                'Scanning for industry keywords...',
                'Analyzing keyword density...',
                'Matching against job requirements...'
            ],
            readabilityAnalysis: [
                'Assessing ATS readability...',
                'Checking parsing compatibility...',
                'Evaluating text structure...',
                'Analyzing readability metrics...'
            ]
        };

        let currentStatusInterval = null;
        let statusMessageIndex = 0;

        /**
         * Enhanced processing status with dynamic rotating messages
         * @param {string} message - Primary status message
         * @param {string} category - Category for dynamic messages (optional)
         * @param {boolean} enableRotation - Whether to rotate through dynamic messages
         */
        function updateProcessingStatus(message, category = null, enableRotation = false) {
            if (!processingStatus) return;
            
            // Clear any existing rotation
            if (currentStatusInterval) {
                clearInterval(currentStatusInterval);
                currentStatusInterval = null;
            }
            
            if (enableRotation && category && DYNAMIC_STATUS_MESSAGES[category]) {
                const messages = DYNAMIC_STATUS_MESSAGES[category];
                statusMessageIndex = 0;
                
                // Start with the primary message
                processingStatus.textContent = message;
                
                // Rotate through dynamic messages every 2 seconds
                currentStatusInterval = setInterval(() => {
                    statusMessageIndex = (statusMessageIndex + 1) % messages.length;
                    processingStatus.textContent = messages[statusMessageIndex];
                    
                    // Add subtle animation effect
                    processingStatus.style.opacity = '0.7';
                    setTimeout(() => {
                        processingStatus.style.opacity = '1';
                    }, 200);
                }, 2000);
            } else {
                // Standard single message
                processingStatus.textContent = message;
            }
        }

        /**
         * Enhanced progress bar with smooth animations and color transitions
         * @param {number} percentage - Progress percentage (0-100)
         * @param {string} stage - Current analysis stage for color coding
         */
        function updateProgress(percentage, stage = 'default') {
            if (!progressBar) return;
            
            // Smooth progress animation
            progressBar.style.transition = 'width 0.5s ease-out, background 0.3s ease';
            progressBar.style.width = `${percentage}%`;
            
            // Dynamic color coding based on stage
            const stageColors = {
                initializing: 'from-blue-400 via-cyan-400 to-blue-500',
                aiAnalysis: 'from-purple-400 via-pink-400 to-purple-500',
                formatAnalysis: 'from-green-400 via-emerald-400 to-green-500',
                contentAnalysis: 'from-rose-400 via-pink-400 to-rose-500',
                keywordAnalysis: 'from-indigo-400 via-purple-400 to-indigo-500',
                readabilityAnalysis: 'from-blue-400 via-purple-400 to-pink-500',
                complete: 'from-green-400 via-emerald-400 to-green-600',
                default: 'from-blue-400 via-purple-400 to-pink-400'
            };
            
            const colorClass = stageColors[stage] || stageColors.default;
            progressBar.className = `bg-gradient-to-r ${colorClass} h-4 rounded-full progress-bar relative`;
            
            // Add pulse effect for active stages
            if (percentage > 0 && percentage < 100) {
                progressBar.style.boxShadow = '0 0 20px rgba(139, 92, 246, 0.5)';
            } else if (percentage === 100) {
                progressBar.style.boxShadow = '0 0 20px rgba(34, 197, 94, 0.5)';
            }
        }

        /**
         * Update score display with dynamic calculation animation
         * @param {number} score - Final score (optional, for final display)
         * @param {string} stage - Current calculation stage
         */
        function updateScoreCalculation(score = null, stage = 'calculating') {
            const scoreGrade = document.getElementById('score-grade');
            if (!scoreGrade) return;
            
            if (score !== null) {
                // Show "Calculation done" first
                scoreGrade.innerHTML = '<span class="gradient-text">Calculation done!</span>';
                
                // After 2 seconds, show the final score
                setTimeout(() => {
                    scoreGrade.textContent = `${score}/100`;
                    scoreGrade.style.color = score >= 80 ? '#06d6a0' : score >= 60 ? '#ffd166' : '#f72585';
                    scoreGrade.style.transform = 'scale(1.1)';
                    setTimeout(() => {
                        scoreGrade.style.transform = 'scale(1)';
                    }, 300);
                }, 2000);
            } else {
                // No message during calculation - keep area empty
                scoreGrade.innerHTML = '';
            }
        }

        /**
         * Clear all dynamic animations and intervals
         */
        function clearDynamicAnimations() {
            if (currentStatusInterval) {
                clearInterval(currentStatusInterval);
                currentStatusInterval = null;
            }
            
            const scoreGrade = document.getElementById('score-grade');
            if (scoreGrade && scoreGrade.dataset.interval) {
                clearInterval(parseInt(scoreGrade.dataset.interval));
                delete scoreGrade.dataset.interval;
            }
        }

        /**
         * Format file size for display
         * @param {number} bytes - File size in bytes
         * @returns {string} Formatted file size
         */
        function formatFileSize(bytes) {
            if (bytes === 0) return '0 Bytes';
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        }

        /**
         * Show error message with improved user feedback
         * @param {string} message - Error message to display
         */
        function showErrorMessage(message) {
            // Create or update error notification
            let errorDiv = document.getElementById('error-notification');
            if (!errorDiv) {
                errorDiv = document.createElement('div');
                errorDiv.id = 'error-notification';
                errorDiv.className = 'fixed top-4 right-4 bg-purple-500 text-white px-6 py-4 rounded-lg shadow-lg z-50 max-w-md';
                document.body.appendChild(errorDiv);
            }
            
            errorDiv.innerHTML = `
                <div class="flex items-center">
                    <svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path>
                    </svg>
                    <span>${message}</span>
                    <button onclick="this.parentElement.parentElement.remove()" class="ml-4 text-white hover:text-gray-200">
                        <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                        </svg>
                    </button>
                </div>
            `;
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                if (errorDiv && errorDiv.parentNode) {
                    errorDiv.remove();
                }
            }, 5000);
        }

        /**
         * Show success message with improved user feedback
         * @param {string} message - Success message to display
         */
        function showSuccessMessage(message) {
            const successDiv = document.createElement('div');
            successDiv.className = 'fixed top-4 right-4 bg-green-500 text-white px-6 py-4 rounded-lg shadow-lg z-50 max-w-md';
            successDiv.innerHTML = `
                <div class="flex items-center">
                    <svg class="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                    </svg>
                    <span>${message}</span>
                    <button onclick="this.remove()" class="ml-4 text-white hover:text-gray-200">
                        <svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
                        </svg>
                    </button>
                </div>
            `;
            
            document.body.appendChild(successDiv);
            
            // Auto-remove after 3 seconds
            setTimeout(() => {
                if (successDiv && successDiv.parentNode) {
                    successDiv.remove();
                }
            }, 3000);
        }

        /**
         * Validate uploaded file with comprehensive error handling
         * @param {File} file - File to validate
         * @returns {boolean} True if file is valid
         */
        function validateFile(file) {
            // Check if file exists
            if (!file) {
                showErrorMessage('No file selected. Please choose a PDF file.');
                return false;
            }

            // Check file type - accept both MIME type and extension
            const isValidType = file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf');
            if (!isValidType) {
                showErrorMessage('Invalid file type. Please upload a PDF file only.');
                return false;
            }

            // Check file size (5MB limit)
            const maxSize = 5 * 1024 * 1024; // 5MB in bytes
            if (file.size > maxSize) {
                const fileSize = formatFileSize(file.size);
                showErrorMessage(`File size (${fileSize}) exceeds the 5MB limit. Please choose a smaller file.`);
                return false;
            }

            // Check minimum file size (avoid empty files)
            if (file.size < 1024) { // 1KB minimum
                showErrorMessage('File appears to be too small or empty. Please upload a valid PDF resume.');
                return false;
            }

            return true;
        }

        /**
         * Handle file selection with comprehensive validation and feedback
         * @param {File} file - Selected file
         */
        /**
         * Handle file selection and automatically start analysis
         * @param {File} file - Selected PDF file
         */
        async function handleFileSelect(file) {
            // Clear any previous error states
            clearPreviousErrors();
            
            if (!file) {
                showErrorMessage('No file selected. Please choose a PDF file to analyze.');
                return;
            }
            
            if (!validateFile(file)) {
                return; // validateFile already shows error message
            }

            try {
                currentFile = file;
                fileName.textContent = file.name;
                fileSize.textContent = formatFileSize(file.size);
                fileInfo.classList.remove('hidden');
                
                // Hide analyze button since we're auto-processing
                analyzeBtn.style.display = 'none';
                
                showSuccessMessage(`File "${file.name}" selected. Starting automatic analysis...`);
                
                // Automatically start analysis after a brief delay
                setTimeout(async () => {
                    try {
                        // Show processing section
                        uploadSection.classList.add('hidden');
                        processingSection.classList.remove('hidden');
                        
                        updateProcessingStatus('Extracting text from PDF...');
                        updateProgress(5);
                        
                        // Extract text from PDF
                        const resumeText = await extractTextFromPDF(currentFile);
                        
                        if (!resumeText.trim()) {
                            throw new Error('No text could be extracted from the PDF. Please ensure the PDF contains readable text.');
                        }

                        updateProgress(20);
                        
                        // Perform ATS analysis
                        await performATSAnalysis(resumeText);
                        
                        // Job matching controls removed
                        
                    } catch (error) {
                        console.error('Analysis error:', error);
                        
                        // Provide specific error messages based on error type
                        let errorMessage = 'An unexpected error occurred during analysis.';
                        
                        if (error.message.includes('No text could be extracted')) {
                            errorMessage = 'Unable to extract text from this PDF. The file may be image-based, corrupted, or password-protected. Please try a different PDF file.';
                        } else if (error.message.includes('Network')) {
                            errorMessage = 'Network error occurred. Please check your internet connection and try again.';
                        } else if (error.message.includes('API')) {
                            errorMessage = 'AI analysis service is temporarily unavailable. The basic analysis will still work.';
                        } else if (error.message) {
                            errorMessage = error.message;
                        }
                        
                        showErrorMessage(errorMessage);
                        resetToUpload();
                    }
                }, 1000); // 1 second delay to show the success message
                
            } catch (error) {
                console.error('File selection error:', error);
                showErrorMessage('An error occurred while processing the selected file. Please try again.');
            }
        }
        
        /**
         * Clear previous error states and messages
         */
        function clearPreviousErrors() {
            // Remove any existing error/success messages
            const existingMessages = document.querySelectorAll('.notification-message');
            existingMessages.forEach(msg => msg.remove());
        }

        /**
         * Remove selected file
         */
        /**
         * Remove selected file and reset UI state
         */
        function removeFile() {
            currentFile = null;
            fileInfo.classList.add('hidden');
            analyzeBtn.style.display = 'block'; // Show analyze button again
            analyzeBtn.disabled = true;
            fileInput.value = '';
        }

        /**
         * Extract text from PDF file
         * @param {File} file - PDF file to extract text from
         * @returns {Promise<string>} Extracted text content
         */
        async function extractTextFromPDF(file) {
            try {
                const arrayBuffer = await file.arrayBuffer();
                const pdf = await pdfjsLib.getDocument(arrayBuffer).promise;
                let fullText = '';

                for (let i = 1; i <= pdf.numPages; i++) {
                    const page = await pdf.getPage(i);
                    const textContent = await page.getTextContent();
                    const pageText = textContent.items.map(item => item.str).join(' ');
                    fullText += pageText + '\n';
                }

                return fullText.trim();
            } catch (error) {
                console.error('Error extracting text from PDF:', error);
                
                // Show animated error notification
                showAnimatedNotification(
                    'Failed to extract text from PDF. Please ensure the file is not corrupted.',
                    'error',
                    5000
                );
                
                throw new Error('Failed to extract text from PDF. Please ensure the file is not corrupted.');
            }
        }

        /**
         * Analyze resume format and structure using optimized scoring
         * @param {string} text - Resume text content
         * @returns {Object} Format analysis results with enhanced scoring
         */
        function analyzeFormat(text) {
            const analysis = {
                score: 0,
                details: [],
                suggestions: [],
                breakdown: {}
            };

            const config = SCORING_CONFIG.formatScoring;
            let contactScore = 0;
            let sectionScore = 0;
            let structureScore = 0;

            // Enhanced contact information analysis
            const emailRegex = /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/;
            const phoneRegex = /\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/;
            const linkedinRegex = /\b(linkedin\.com\/in\/|linkedin\.com\/pub\/)/i;
            
            if (emailRegex.test(text)) {
                contactScore += config.contactInfo.email;
                analysis.details.push('✓ Professional email address found');
            } else {
                analysis.suggestions.push('Add a professional email address');
            }

            if (phoneRegex.test(text)) {
                contactScore += config.contactInfo.phone;
                analysis.details.push('✓ Phone number found');
            } else {
                analysis.suggestions.push('Include a phone number');
            }

            if (linkedinRegex.test(text)) {
                contactScore += config.contactInfo.linkedin;
                analysis.details.push('✓ LinkedIn profile included');
            } else {
                analysis.suggestions.push('Consider adding LinkedIn profile URL');
            }

            // Enhanced section analysis
            const sections = [
                { 
                    name: 'Experience', 
                    regex: /\b(experience|work history|employment|professional experience|career history)\b/i, 
                    points: config.sections.experience 
                },
                { 
                    name: 'Education', 
                    regex: /\b(education|academic|degree|university|college|certification)\b/i, 
                    points: config.sections.education 
                },
                { 
                    name: 'Skills', 
                    regex: /\b(skills|competencies|technical skills|core competencies|expertise)\b/i, 
                    points: config.sections.skills 
                }
            ];

            sections.forEach(section => {
                if (section.regex.test(text)) {
                    sectionScore += section.points;
                    analysis.details.push(`✓ ${section.name} section found`);
                } else {
                    analysis.suggestions.push(`Add a ${section.name} section`);
                }
            });
            }

            // Enhanced structure analysis
            const headingRegex = /^[A-Z][A-Z\s]+$/gm; // All caps headings
            const headingMatches = text.match(headingRegex);
            if (headingMatches && headingMatches.length >= 3) {
                structureScore += config.structure.clearHeadings;
                analysis.details.push('✓ Clear section headings detected');
            } else {
                analysis.suggestions.push('Use clear, consistent section headings');
            }

            // Check for consistent formatting (bullet points, dates, etc.)
            const bulletRegex = /[•·▪▫-]\s/g;
            const dateRegex = /\b(19|20)\d{2}\b/g;
            const bulletMatches = text.match(bulletRegex);
            const dateMatches = text.match(dateRegex);
            
            if (bulletMatches && bulletMatches.length >= 3 && dateMatches && dateMatches.length >= 2) {
                structureScore += config.structure.consistentFormatting;
                analysis.details.push('✓ Consistent formatting detected');
            } else {
                analysis.suggestions.push('Use consistent formatting for dates and bullet points');
            }

            // Calculate final score and breakdown
            analysis.breakdown = {
                contactInfo: contactScore,
                sections: sectionScore,
                structure: structureScore
            };
            
            analysis.score = Math.min(contactScore + sectionScore + structureScore, 100);
            
            // Add comprehensive profile bonus
            if (contactScore >= 35 && sectionScore >= 60 && structureScore >= 15) {
                const bonus = SCORING_CONFIG.bonusScoring.comprehensiveProfile;
                analysis.score = Math.min(analysis.score + bonus, 100);
                analysis.details.push('✓ Comprehensive profile bonus applied');
            }

            return analysis;
        }

        /**
         * Analyze resume content quality using optimized scoring
         * @param {string} text - Resume text content
         * @param {Object} aiAnalysis - AI analysis results (optional)
         * @returns {Object} Enhanced content analysis results
         */
        function analyzeContent(text, aiAnalysis = null) {
            const analysis = {
                score: 0,
                details: [],
                suggestions: [],
                breakdown: {}
            };

            const config = SCORING_CONFIG.contentScoring;
            let wordCountScore = 0;
            let actionVerbScore = 0;
            let achievementScore = 0;
            let aiEnhancementScore = 0;

            // Enhanced word count analysis
            const wordCount = text.split(/\s+/).filter(word => word.length > 0).length;
            if (wordCount >= config.wordCount.optimal) {
                wordCountScore = config.wordCount.points;
                analysis.details.push(`✓ Optimal content length (${wordCount} words)`);
            } else if (wordCount >= config.wordCount.minimum) {
                wordCountScore = Math.round((wordCount / config.wordCount.optimal) * config.wordCount.points);
                analysis.details.push(`✓ Good content length (${wordCount} words)`);
            } else {
                analysis.suggestions.push(`Expand content to at least ${config.wordCount.minimum} words (currently ${wordCount})`);
            }

            // Enhanced action verb analysis
            const actionVerbs = [
                'achieved', 'managed', 'led', 'developed', 'created', 'implemented', 
                'improved', 'increased', 'reduced', 'designed', 'optimized', 'streamlined',
                'coordinated', 'executed', 'delivered', 'established', 'initiated', 
                'transformed', 'enhanced', 'accelerated', 'generated', 'facilitated'
            ];
            const foundVerbs = actionVerbs.filter(verb => new RegExp(`\\b${verb}`, 'i').test(text));
            
            if (foundVerbs.length >= config.actionVerbs.optimal) {
                actionVerbScore = config.actionVerbs.points;
                analysis.details.push(`✓ Excellent use of action verbs (${foundVerbs.length} found)`);
            } else if (foundVerbs.length >= config.actionVerbs.minimum) {
                actionVerbScore = Math.round((foundVerbs.length / config.actionVerbs.optimal) * config.actionVerbs.points);
                analysis.details.push(`✓ Good use of action verbs (${foundVerbs.length} found)`);
            } else {
                analysis.suggestions.push(`Use more action verbs to describe achievements (found ${foundVerbs.length}, need at least ${config.actionVerbs.minimum})`);
            }

            // Enhanced quantifiable achievements analysis
            const numberRegex = /\b\d+(%|\$|k|million|billion|years?|months?|days?|hours?|projects?|clients?|teams?|people|employees|revenue|sales|growth|reduction|improvement)\b/gi;
            const percentageRegex = /\b\d+%\b/g;
            const currencyRegex = /\$[\d,]+/g;
            
            const numbers = text.match(numberRegex) || [];
            const percentages = text.match(percentageRegex) || [];
            const currency = text.match(currencyRegex) || [];
            
            const totalMetrics = numbers.length + percentages.length + currency.length;
            
            if (totalMetrics >= config.quantifiableAchievements.optimal) {
                achievementScore = config.quantifiableAchievements.points;
                analysis.details.push(`✓ Excellent quantifiable achievements (${totalMetrics} metrics found)`);
            } else if (totalMetrics >= config.quantifiableAchievements.minimum) {
                achievementScore = Math.round((totalMetrics / config.quantifiableAchievements.optimal) * config.quantifiableAchievements.points);
                analysis.details.push(`✓ Good quantifiable achievements (${totalMetrics} metrics found)`);
            } else {
                analysis.suggestions.push(`Add more quantifiable achievements and metrics (found ${totalMetrics}, need at least ${config.quantifiableAchievements.minimum})`);
            }

            // Enhanced AI analysis integration
            if (aiAnalysis) {
                if (aiAnalysis.strengths && aiAnalysis.strengths.length > 0) {
                    aiEnhancementScore = config.aiEnhancement;
                    analysis.details.push('🤖 AI-enhanced content analysis completed');
                    analysis.details.push(`🤖 AI identified ${aiAnalysis.strengths.length} key strengths`);
                }
                
                // Add AI-specific suggestions with priority
                if (aiAnalysis.improvements && aiAnalysis.improvements.length > 0) {
                    const priorityImprovements = aiAnalysis.improvements.slice(0, 2);
                    analysis.suggestions.push(...priorityImprovements.map(imp => `🤖 AI Suggestion: ${imp}`));
                }
                
                // Bonus for comprehensive AI analysis
                if (aiAnalysis.overallScore && aiAnalysis.overallScore >= 70) {
                    aiEnhancementScore += 5;
                    analysis.details.push('🤖 High-quality content detected by AI');
                }
            } else {
                analysis.suggestions.push('Enable AI analysis for more detailed content insights');
            }

            // Calculate final score and breakdown
            analysis.breakdown = {
                wordCount: wordCountScore,
                actionVerbs: actionVerbScore,
                achievements: achievementScore,
                aiEnhancement: aiEnhancementScore
            };
            
            analysis.score = Math.min(wordCountScore + actionVerbScore + achievementScore + aiEnhancementScore, 100);
            
            return analysis;
        }

        /**
         * Analyze keyword optimization with enhanced SCORING_CONFIG integration
         * @param {string} text - Resume text content
         * @param {Object} aiAnalysis - AI analysis results (optional)
         * @returns {Object} Keyword analysis results with detailed scoring breakdown
         */
        function analyzeKeywords(text, aiAnalysis = null) {
            const analysis = {
                score: 0,
                details: [],
                suggestions: [],
                breakdown: {}
            };
            
            const config = SCORING_CONFIG.keywordScoring;

            // Enhanced professional keywords categorized by importance
            const keywords = {
                core: ['leadership', 'management', 'project', 'team', 'analysis', 'strategy'],
                technical: ['development', 'implementation', 'optimization', 'innovation'],
                soft: ['collaboration', 'communication', 'problem-solving', 'results']
            };

            // Calculate keyword coverage scores
            const coreFound = keywords.core.filter(keyword => 
                new RegExp(`\\b${keyword}`, 'i').test(text)
            );
            const technicalFound = keywords.technical.filter(keyword => 
                new RegExp(`\\b${keyword}`, 'i').test(text)
            );
            const softFound = keywords.soft.filter(keyword => 
                new RegExp(`\\b${keyword}`, 'i').test(text)
            );

            // Calculate weighted scores based on SCORING_CONFIG
            const coreScore = Math.min((coreFound.length / keywords.core.length) * config.coreKeywords.maxScore, config.coreKeywords.maxScore);
            const technicalScore = Math.min((technicalFound.length / keywords.technical.length) * config.technicalKeywords.maxScore, config.technicalKeywords.maxScore);
            const softScore = Math.min((softFound.length / keywords.soft.length) * config.softSkills.maxScore, config.softSkills.maxScore);
            
            let baseScore = Math.round(coreScore + technicalScore + softScore);
            
            // Store breakdown for detailed analysis
            analysis.breakdown = {
                coreKeywords: coreScore,
                technicalKeywords: technicalScore,
                softSkills: softScore,
                aiEnhancement: 0
            };
            
            // Enhanced AI analysis integration
            if (aiAnalysis && aiAnalysis.keywordOptimization) {
                // Use optimized AI weighting from SCORING_CONFIG