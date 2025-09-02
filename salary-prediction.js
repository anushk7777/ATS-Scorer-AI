/**
 * Salary Prediction Module
 * 
 * This module provides client-side salary prediction functionality for the ATS application.
 * It extracts features from resume analysis and predicts salary using the trained ML model.
 * 
 * Features:
 * - Feature extraction from resume text and ATS analysis
 * - Client-side salary prediction using simplified model
 * - Integration with existing ATS scoring system
 * - Confidence scoring and validation
 * 
 * Author: AI Assistant
 * Date: 2025
 */

class SalaryPredictor {
    /**
     * Initialize the salary predictor with model configuration
     */
    constructor() {
        this.modelConfig = null;
        this.isInitialized = false;
        this.featureExtractor = new ResumeFeatureExtractor();
        this.useBackendService = true; // Enable backend service by default
        
        // Load model configuration
        this.loadModelConfig();
    }
    
    /**
     * Load the trained model configuration from JSON file
     */
    async loadModelConfig() {
        try {
            const response = await fetch('./salary_model_config.json');
            this.modelConfig = await response.json();
            this.isInitialized = true;
            console.log('Salary prediction model loaded successfully');
            console.log(`Model type: ${this.modelConfig.model_type}`);
            console.log(`Model R²: ${this.modelConfig.performance_metrics.test_r2.toFixed(4)}`);
        } catch (error) {
            console.error('Failed to load salary prediction model:', error);
            this.isInitialized = false;
        }
    }
    
    /**
     * Extract salary prediction features from resume analysis results
     * 
     * @param {Object} resumeAnalysis - Results from ATS analysis
     * @param {string} resumeText - Raw resume text
     * @returns {Object} Extracted features for salary prediction
     */
    extractFeatures(resumeAnalysis, resumeText) {
        if (!this.isInitialized) {
            throw new Error('Salary predictor not initialized');
        }
        
        return this.featureExtractor.extract(resumeAnalysis, resumeText);
    }
    
    /**
     * Predict salary based on extracted features
     * 
     * @param {Object} features - Extracted resume features
     * @param {string} resumeText - Optional resume text for backend service
     * @returns {Promise<Object>} Salary prediction with confidence and details
     */
    async predictSalary(features, resumeText = '') {
        if (!this.isInitialized) {
            throw new Error('Salary predictor not initialized');
        }
        
        try {
            // Encode categorical features
            const encodedFeatures = this.encodeFeatures(features);
            
            // Make prediction using enhanced model
            const prediction = await this.makePrediction(encodedFeatures, resumeText);
            
            // Calculate confidence score
            const confidence = this.calculateConfidence(features, prediction);
            
            // Format result
            return {
                salary: {
                    lakhs: Math.round(prediction * 10) / 10,
                    rupees: Math.round(prediction * 100000),
                    formatted: `₹${(prediction * 100000).toLocaleString('en-IN')}`
                },
                confidence: confidence,
                range: {
                    min: Math.round((prediction * 0.85) * 100000),
                    max: Math.round((prediction * 1.15) * 100000)
                },
                factors: this.getInfluencingFactors(features),
                recommendations: this.generateRecommendations(features, prediction)
            };
        } catch (error) {
            console.error('Salary prediction failed:', error);
            return this.getDefaultPrediction();
        }
    }
    
    /**
     * Encode categorical features using label encoders from model config
     * 
     * @param {Object} features - Raw features
     * @returns {Array} Encoded feature vector
     */
    encodeFeatures(features) {
        const encodedVector = [];
        
        for (const featureName of this.modelConfig.feature_names) {
            let value = features[featureName] || 0;
            
            // Apply label encoding for categorical features
            if (this.modelConfig.label_encoders[featureName]) {
                const encoder = this.modelConfig.label_encoders[featureName];
                const classes = encoder.classes;
                
                // Find the index of the value in classes
                const index = classes.indexOf(value);
                value = index >= 0 ? index : 0; // Default to 0 if not found
            }
            
            encodedVector.push(value);
        }
        
        return encodedVector;
    }
    
    /**
     * Enhanced salary prediction with backend integration
     * Considers organizational salary bands, internal equity, and experience-specific factors
     * 
     * @param {Array} encodedFeatures - Encoded feature vector
     * @param {string} resumeText - Optional resume text for backend service
     * @returns {Promise<number>} Predicted salary in lakhs
     */
    async makePrediction(encodedFeatures, resumeText = '') {
        try {
            // Try to use enhanced backend service first
            if (resumeText && this.useBackendService) {
                const backendResult = await this.callBackendPrediction(resumeText);
                if (backendResult && !backendResult.error) {
                    return backendResult.predicted_salary;
                }
            }
            
            // Fallback to client-side prediction
            return this.clientSidePrediction(encodedFeatures);
            
        } catch (error) {
            console.error('Error in salary prediction:', error);
            return this.clientSidePrediction(encodedFeatures);
        }
    }
    
    /**
     * Client-side prediction (original algorithm)
     * 
     * @param {Array} encodedFeatures - Encoded feature vector
     * @returns {number} Predicted salary in lakhs
     */
    clientSidePrediction(encodedFeatures) {
        const featureNames = this.modelConfig.feature_names;
        const features = {};
        
        // Map encoded features back to feature names
        featureNames.forEach((name, index) => {
            features[name] = encodedFeatures[index];
        });
        
        // Enhanced salary calculation for HR system
        let baseSalary = this.calculateBaseSalaryByExperience(features);
        
        // Apply role-specific adjustments with updated multipliers
        baseSalary = this.applyRoleAdjustments(baseSalary, features);
        
        // Apply location-based cost of living adjustments
        baseSalary = this.applyLocationAdjustments(baseSalary, features);
        
        // Apply performance and skill-based premiums
        baseSalary = this.applyPerformanceAdjustments(baseSalary, features);
        
        // Apply internal equity and market positioning
        baseSalary = this.applyMarketPositioning(baseSalary, features);
        
        // Ensure salary falls within organizational bands
        return this.enforceOrganizationalBands(baseSalary, features);
    }
    
    /**
     * Call enhanced backend service for salary prediction
     * 
     * @param {string} resumeText - Resume text
     * @returns {Promise<Object>} Backend prediction result
     */
    async callBackendPrediction(resumeText) {
        try {
            const response = await fetch('/api/predict-salary', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    resume_text: resumeText
                })
            });
            
            if (!response.ok) {
                throw new Error(`Backend service error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.warn('Backend service unavailable, using client-side prediction:', error);
            return null;
        }
    }
    
    /**
     * Calculate base salary based on experience level with improved logic
     * 
     * @param {Object} features - Feature object
     * @returns {number} Base salary in lakhs
     */
    calculateBaseSalaryByExperience(features) {
        const experienceGroup = features.experience_group;
        const yearsOfExperience = features.years_of_experience || 0;
        
        // Enhanced base salary calculation by experience group
        switch(experienceGroup) {
            case 0: // Freshers (0-1 years)
                // For freshers, focus on education, skills, and potential
                return this.calculateFresherSalary(features);
                
            case 1: // Junior (1-3 years)
                // Base salary with experience progression
                return 8.5 + (yearsOfExperience * 1.2);
                
            case 2: // Mid-level (3-7 years)
                // Significant experience premium with skill specialization
                return 12.0 + (yearsOfExperience * 1.8);
                
            case 3: // Senior (7+ years)
                // Leadership and strategic impact premium
                return 18.0 + (Math.min(yearsOfExperience, 15) * 2.2);
                
            default:
                return 8.0;
        }
    }
    
    /**
     * Calculate salary for freshers based on education, skills, and potential
     * 
     * @param {Object} features - Feature object
     * @returns {number} Fresher base salary in lakhs
     */
    calculateFresherSalary(features) {
        let fresherBase = 6.5; // Conservative base for freshers
        
        // Education level premium for freshers
        const educationLevel = features.education_level || 0;
        const educationPremiums = {
            0: 0.0,   // High School
            1: 0.5,   // Diploma
            2: 1.0,   // Bachelor's
            3: 2.0,   // Master's
            4: 3.0    // PhD
        };
        fresherBase += educationPremiums[educationLevel] || 0;
        
        // Skills and potential assessment (based on resume quality)
        const performanceRating = features.performance_rating || 3.0;
        if (performanceRating > 4.0) {
            fresherBase += 1.5; // High potential candidate
        } else if (performanceRating > 3.5) {
            fresherBase += 0.8; // Good candidate
        }
        
        // Technical skills bonus for freshers
        const crossFunctionalExp = features.cross_functional_experience || 0;
        fresherBase += Math.min(crossFunctionalExp * 0.3, 1.0);
        
        return fresherBase;
    }
    
    /**
     * Apply role-specific salary adjustments with updated multipliers
     * 
     * @param {number} baseSalary - Base salary
     * @param {Object} features - Feature object
     * @returns {number} Adjusted salary
     */
    applyRoleAdjustments(baseSalary, features) {
        // Updated role multipliers based on market demand and organizational needs
        const roleMultipliers = {
            0: 1.0,   // Backend Developer
            1: 1.35,  // Data Scientist (high demand)
            2: 1.25,  // DevOps Engineer (critical role)
            3: 0.95,  // Frontend Developer
            4: 1.15,  // Full-stack Developer (versatile)
            5: 1.0,   // Mobile Developer
            6: 1.45,  // Product Manager (strategic role)
            7: 0.85,  // QA Engineer
            8: 1.6,   // Tech Lead (leadership premium)
            9: 0.9    // UI/UX Designer
        };
        
        const roleMultiplier = roleMultipliers[features.role] || 1.0;
        return baseSalary * roleMultiplier;
    }
    
    /**
     * Apply location-based cost of living adjustments
     * 
     * @param {number} baseSalary - Base salary
     * @param {Object} features - Feature object
     * @returns {number} Adjusted salary
     */
    applyLocationAdjustments(baseSalary, features) {
        // Updated location multipliers based on cost of living and market rates
        const locationMultipliers = {
            0: 1.25,  // Bangalore (tech hub premium)
            1: 0.9,   // Chennai
            2: 1.15,  // Delhi
            3: 1.2,   // Gurgaon (corporate hub)
            4: 1.05,  // Hyderabad
            5: 1.4,   // Mumbai (highest cost of living)
            6: 1.15,  // Noida
            7: 1.0    // Pune (baseline)
        };
        
        const locationMultiplier = locationMultipliers[features.location] || 1.0;
        return baseSalary * locationMultiplier;
    }
    
    /**
     * Apply performance and skill-based salary premiums
     * 
     * @param {number} baseSalary - Base salary
     * @param {Object} features - Feature object
     * @returns {number} Adjusted salary
     */
    applyPerformanceAdjustments(baseSalary, features) {
        let adjustedSalary = baseSalary;
        
        // Performance rating impact (more significant for experienced professionals)
        if (features.performance_rating) {
            const performanceMultiplier = 0.7 + (features.performance_rating * 0.08);
            adjustedSalary *= performanceMultiplier;
        }
        
        // Leadership score impact (especially for mid-level and senior)
        if (features.leadership_score && features.experience_group >= 1) {
            const leadershipBonus = 1.0 + (features.leadership_score * 0.015);
            adjustedSalary *= leadershipBonus;
        }
        
        // Cross-functional experience premium
        if (features.cross_functional_experience) {
            const crossFunctionalBonus = 1.0 + (features.cross_functional_experience * 0.025);
            adjustedSalary *= crossFunctionalBonus;
        }
        
        return adjustedSalary;
    }
    
    /**
     * Apply market positioning strategy (competitive vs conservative)
     * 
     * @param {number} baseSalary - Base salary
     * @param {Object} features - Feature object
     * @returns {number} Market-positioned salary
     */
    applyMarketPositioning(baseSalary, features) {
        // Use market rate as reference but apply organizational strategy
        if (features.market_rate) {
            const marketRate = features.market_rate;
            
            // Position salary at 85-95% of market rate based on experience
            const marketPositioning = {
                0: 0.85, // Freshers: 85% of market (training investment)
                1: 0.88, // Junior: 88% of market
                2: 0.92, // Mid-level: 92% of market
                3: 0.95  // Senior: 95% of market (retention focus)
            };
            
            const positioningFactor = marketPositioning[features.experience_group] || 0.90;
            const marketBasedSalary = marketRate * positioningFactor;
            
            // Take the higher of calculated salary or market-based salary
            return Math.max(baseSalary, marketBasedSalary);
        }
        
        return baseSalary;
    }
    
    /**
     * Enforce organizational salary bands and ensure internal equity
     * 
     * @param {number} salary - Calculated salary
     * @param {Object} features - Feature object
     * @returns {number} Final salary within organizational bands
     */
    enforceOrganizationalBands(salary, features) {
        // Define salary bands by experience group
        const salaryBands = {
            0: { min: 5.0, max: 12.0 },  // Freshers
            1: { min: 8.0, max: 18.0 },  // Junior
            2: { min: 15.0, max: 30.0 }, // Mid-level
            3: { min: 25.0, max: 50.0 }  // Senior
        };
        
        const experienceGroup = features.experience_group || 0;
        const band = salaryBands[experienceGroup] || { min: 5.0, max: 35.0 };
        
        // Ensure salary falls within the band
        return Math.max(band.min, Math.min(band.max, salary));
    }
    
    /**
     * Calculate confidence score for the prediction
     * 
     * @param {Object} features - Raw features
     * @param {number} prediction - Predicted salary
     * @returns {number} Confidence score (0-100)
     */
    calculateConfidence(features, prediction) {
        let confidence = 70; // Base confidence
        
        // Higher confidence if market rate is available
        if (features.market_rate && features.market_rate > 0) {
            confidence += 20;
        }
        
        // Higher confidence for standard roles and locations
        const standardRoles = ['Backend Developer', 'Frontend Developer', 'Full-stack Developer'];
        if (standardRoles.includes(features.role)) {
            confidence += 5;
        }
        
        const majorCities = ['Mumbai', 'Bangalore', 'Delhi', 'Gurgaon'];
        if (majorCities.includes(features.location)) {
            confidence += 5;
        }
        
        // Reduce confidence for extreme predictions
        if (prediction < 6 || prediction > 30) {
            confidence -= 15;
        }
        
        return Math.max(50, Math.min(95, confidence));
    }
    
    /**
     * Match resume with job description using enhanced backend service
     * 
     * @param {string} resumeText - Resume text content
     * @param {string} jobDescription - Job description text
     * @returns {Promise<Object>} Job matching analysis
     */
    async matchJobDescription(resumeText, jobDescription) {
        try {
            // Try backend service first
            if (this.useBackendService) {
                const backendResult = await this.callJobMatchingService(resumeText, jobDescription);
                if (backendResult && !backendResult.error) {
                    return backendResult;
                }
            }
            
            // Fallback to client-side matching
            return this.clientSideJobMatching(resumeText, jobDescription);
            
        } catch (error) {
            console.error('Error in job matching:', error);
            return this.clientSideJobMatching(resumeText, jobDescription);
        }
    }
    
    /**
     * Call backend job matching service
     * 
     * @param {string} resumeText - Resume text
     * @param {string} jobDescription - Job description
     * @returns {Promise<Object>} Backend matching result
     */
    async callJobMatchingService(resumeText, jobDescription) {
        try {
            const response = await fetch('/api/match-job', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    resume_text: resumeText,
                    job_description: jobDescription
                })
            });
            
            if (!response.ok) {
                throw new Error(`Backend service error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.warn('Job matching service unavailable, using client-side matching:', error);
            return null;
        }
    }
    

    
    /**
     * Extract skills from job description
     * 
     * @param {string} jobDescription - Job description text
     * @returns {Array<string>} Extracted skills
     */
    extractJobSkills(jobDescription) {
        const skillKeywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js',
            'django', 'flask', 'spring', 'sql', 'mongodb', 'postgresql',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git',
            'machine learning', 'data science', 'artificial intelligence',
            'html', 'css', 'typescript', 'c++', 'c#', 'go', 'rust'
        ];
        
        const text = jobDescription.toLowerCase();
        return skillKeywords.filter(skill => text.includes(skill));
    }
    
    /**
     * Extract skills from resume text
     * 
     * @param {string} resumeText - Resume text
     * @returns {Array<string>} Extracted skills
     */
    extractResumeSkills(resumeText) {
        const skillKeywords = [
            'python', 'java', 'javascript', 'react', 'angular', 'vue', 'node.js',
            'django', 'flask', 'spring', 'sql', 'mongodb', 'postgresql',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git',
            'machine learning', 'data science', 'artificial intelligence',
            'html', 'css', 'typescript', 'c++', 'c#', 'go', 'rust'
        ];
        
        const text = resumeText.toLowerCase();
        return skillKeywords.filter(skill => text.includes(skill));
    }
    
    /**
     * Extract required experience from job description
     * 
     * @param {string} jobDescription - Job description text
     * @returns {string} Required experience
     */
    extractRequiredExperience(jobDescription) {
        const expMatch = jobDescription.match(/(\d+)\+?\s*years?\s*of\s*experience/i);
        return expMatch ? `${expMatch[1]} years` : 'Not specified';
    }
    
    /**
     * Evaluate experience match
     * 
     * @param {Object} resumeFeatures - Resume features
     * @param {string} jobDescription - Job description
     * @returns {number} Experience match percentage
     */
    evaluateExperienceMatch(resumeFeatures, jobDescription) {
        const requiredExp = this.extractRequiredExperience(jobDescription);
        if (requiredExp === 'Not specified') return 100;
        
        const requiredYears = parseInt(requiredExp);
        const candidateYears = resumeFeatures.years_of_experience;
        
        if (candidateYears >= requiredYears) return 100;
        if (candidateYears >= requiredYears * 0.8) return 80;
        if (candidateYears >= requiredYears * 0.6) return 60;
        return 40;
    }
    
    /**
     * Generate recommendations for improving job match
     * 
     * @param {Array<string>} matchedSkills - Matched skills
     * @param {Array<string>} missingSkills - Missing skills
     * @returns {Array<string>} Recommendations
     */
    generateJobMatchRecommendations(matchedSkills, missingSkills) {
        const recommendations = [];
        
        if (missingSkills.length > 0) {
            recommendations.push(`Consider learning: ${missingSkills.slice(0, 3).join(', ')}`);
        }
        
        if (matchedSkills.length < 3) {
            recommendations.push('Strengthen technical skills portfolio');
        }
        
        recommendations.push('Highlight relevant project experience');
        
        return recommendations;
    }
    
    /**
     * Assess overall fit for position
     * 
     * @param {number} matchScore - Match score (0-1)
     * @returns {string} Fit assessment
     */
    assessFit(matchScore) {
        if (matchScore >= 0.8) return 'Excellent fit';
        if (matchScore >= 0.6) return 'Good fit';
        if (matchScore >= 0.4) return 'Moderate fit';
        return 'Limited fit';
    }
    
    /**
     * Get factors that most influenced the salary prediction
     * 
     * @param {Object} features - Raw features
     * @returns {Array} Array of influencing factors
     */
    getInfluencingFactors(features) {
        const factors = [];
        
        if (features.market_rate) {
            factors.push({
                factor: 'Market Rate',
                value: `₹${features.market_rate} lakhs`,
                impact: 'High',
                description: 'Primary factor in salary determination'
            });
        }
        
        if (features.years_of_experience) {
            factors.push({
                factor: 'Experience',
                value: `${features.years_of_experience} years`,
                impact: 'Medium',
                description: 'Experience level affects salary progression'
            });
        }
        
        if (features.role) {
            factors.push({
                factor: 'Role',
                value: features.role,
                impact: 'Medium',
                description: 'Role type influences compensation'
            });
        }
        
        if (features.location) {
            factors.push({
                factor: 'Location',
                value: features.location,
                impact: 'Medium',
                description: 'Geographic location affects salary levels'
            });
        }
        
        return factors;
    }
    
    /**
     * Generate salary improvement recommendations
     * 
     * @param {Object} features - Raw features
     * @param {number} prediction - Predicted salary
     * @returns {Array} Array of recommendations
     */
    generateRecommendations(features, prediction) {
        const recommendations = [];
        
        // Experience-based recommendations
        if (features.years_of_experience < 3) {
            recommendations.push({
                category: 'Experience',
                suggestion: 'Gain more hands-on experience in your domain',
                impact: 'High',
                timeframe: '1-2 years'
            });
        }
        
        // Skill-based recommendations
        if (features.performance_rating < 4.0) {
            recommendations.push({
                category: 'Performance',
                suggestion: 'Focus on improving performance metrics and deliverables',
                impact: 'High',
                timeframe: '6-12 months'
            });
        }
        
        // Leadership recommendations
        if (features.leadership_score < 7.0 && features.years_of_experience > 2) {
            recommendations.push({
                category: 'Leadership',
                suggestion: 'Develop leadership skills and take on team responsibilities',
                impact: 'Medium',
                timeframe: '1-2 years'
            });
        }
        
        // Cross-functional experience
        if (features.cross_functional_experience < 2) {
            recommendations.push({
                category: 'Versatility',
                suggestion: 'Gain experience in cross-functional projects',
                impact: 'Medium',
                timeframe: '6-18 months'
            });
        }
        
        return recommendations;
    }
    
    /**
     * Get default prediction when actual prediction fails
     * 
     * @returns {Object} Default prediction object
     */
    getDefaultPrediction() {
        return {
            salary: {
                lakhs: 12.0,
                rupees: 1200000,
                formatted: '₹12,00,000'
            },
            confidence: 50,
            range: {
                min: 1000000,
                max: 1400000
            },
            factors: [{
                factor: 'Default Estimate',
                value: 'Industry Average',
                impact: 'Medium',
                description: 'Based on general market trends'
            }],
            recommendations: [{
                category: 'General',
                suggestion: 'Provide more detailed resume information for better prediction',
                impact: 'High',
                timeframe: 'Immediate'
            }]
        };
    }
}

/**
 * Resume Feature Extractor
 * 
 * Extracts ML features from resume text and ATS analysis results
 */
class ResumeFeatureExtractor {
    constructor() {
        this.roleKeywords = {
            'Backend Developer': ['backend', 'server', 'api', 'database', 'node', 'python', 'java', 'spring'],
            'Frontend Developer': ['frontend', 'react', 'angular', 'vue', 'javascript', 'css', 'html', 'ui'],
            'Full-stack Developer': ['fullstack', 'full-stack', 'frontend', 'backend', 'react', 'node'],
            'Data Scientist': ['data science', 'machine learning', 'python', 'pandas', 'tensorflow', 'analytics'],
            'DevOps Engineer': ['devops', 'docker', 'kubernetes', 'aws', 'jenkins', 'ci/cd', 'terraform'],
            'Mobile Developer': ['mobile', 'android', 'ios', 'react native', 'flutter', 'swift', 'kotlin'],
            'Product Manager': ['product management', 'roadmap', 'strategy', 'stakeholder', 'agile'],
            'QA Engineer': ['testing', 'qa', 'automation', 'selenium', 'quality assurance'],
            'Tech Lead': ['tech lead', 'technical lead', 'architecture', 'team lead', 'mentoring'],
            'UI/UX Designer': ['ui', 'ux', 'design', 'figma', 'sketch', 'user experience', 'wireframe']
        };
        
        this.locationKeywords = {
            'Mumbai': ['mumbai', 'bombay'],
            'Bangalore': ['bangalore', 'bengaluru'],
            'Delhi': ['delhi', 'new delhi'],
            'Gurgaon': ['gurgaon', 'gurugram'],
            'Noida': ['noida'],
            'Pune': ['pune'],
            'Hyderabad': ['hyderabad'],
            'Chennai': ['chennai', 'madras']
        };
    }
    
    /**
     * Extract comprehensive features from resume analysis and text for HR system
     * Enhanced to better handle all experience levels and organizational needs
     * 
     * @param {Object} resumeAnalysis - ATS analysis results
     * @param {string} resumeText - Raw resume text
     * @returns {Object} Extracted features with enhanced accuracy
     */
    extract(resumeAnalysis, resumeText) {
        const features = {};
        
        // Extract experience information
        features.years_of_experience = this.extractExperience(resumeText);
        features.experience_group = this.categorizeExperience(features.years_of_experience);
        
        // Extract role information
        features.role = this.extractRole(resumeText, resumeAnalysis);
        
        // Extract location information
        features.location = this.extractLocation(resumeText);
        
        // Extract education level with enhanced detection
        features.education_level = this.extractEducation(resumeText);
        
        // Estimate market rate based on role and experience
        features.market_rate = this.estimateMarketRate(features.role, features.years_of_experience, features.location);
        
        // Extract performance indicators
        features.performance_rating = this.estimatePerformance(resumeAnalysis, resumeText);
        
        // Extract leadership indicators
        features.leadership_score = this.extractLeadershipScore(resumeText);
        
        // Extract cross-functional experience
        features.cross_functional_experience = this.extractCrossFunctionalExperience(resumeText);
        
        // Additional features for better prediction accuracy
        features.technical_skills_score = this.assessTechnicalSkills(resumeText);
        features.project_complexity_score = this.assessProjectComplexity(resumeText);
        features.certification_score = this.assessCertifications(resumeText);
        
        // Adjust features based on experience group for better accuracy
        return this.adjustFeaturesForExperienceGroup(features, resumeText);
    }
    
    /**
     * Extract years of experience from resume text
     * 
     * @param {string} resumeText - Resume text
     * @returns {number} Years of experience
     */
    extractExperience(resumeText) {
        const text = resumeText.toLowerCase();
        
        // Look for experience patterns
        const patterns = [
            /(\d+)\+?\s*years?\s*of\s*experience/,
            /(\d+)\+?\s*years?\s*experience/,
            /experience\s*:?\s*(\d+)\+?\s*years?/,
            /(\d+)\+?\s*yrs?\s*experience/
        ];
        
        for (const pattern of patterns) {
            const match = text.match(pattern);
            if (match) {
                return parseInt(match[1]);
            }
        }
        
        // Fallback: count job positions and estimate
        const jobCount = (text.match(/\b(software engineer|developer|analyst|manager)\b/g) || []).length;
        return Math.min(jobCount * 1.5, 10); // Estimate based on job count
    }
    
    /**
     * Categorize experience level
     * 
     * @param {number} years - Years of experience
     * @returns {string} Experience category
     */
    categorizeExperience(years) {
        if (years <= 1) return 'freshers';
        if (years <= 3) return 'junior';
        if (years <= 7) return 'mid_level';
        return 'senior';
    }
    
    /**
     * Extract role from resume text and analysis
     * 
     * @param {string} resumeText - Resume text
     * @param {Object} resumeAnalysis - ATS analysis
     * @returns {string} Detected role
     */
    extractRole(resumeText, resumeAnalysis) {
        const text = resumeText.toLowerCase();
        
        // Score each role based on keyword matches
        const roleScores = {};
        
        for (const [role, keywords] of Object.entries(this.roleKeywords)) {
            let score = 0;
            for (const keyword of keywords) {
                const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
                const matches = text.match(regex);
                if (matches) {
                    score += matches.length;
                }
            }
            roleScores[role] = score;
        }
        
        // Find the role with highest score
        const bestRole = Object.keys(roleScores).reduce((a, b) => 
            roleScores[a] > roleScores[b] ? a : b
        );
        
        return roleScores[bestRole] > 0 ? bestRole : 'Backend Developer'; // Default
    }
    
    /**
     * Extract location from resume text
     * 
     * @param {string} resumeText - Resume text
     * @returns {string} Detected location
     */
    extractLocation(resumeText) {
        const text = resumeText.toLowerCase();
        
        for (const [location, keywords] of Object.entries(this.locationKeywords)) {
            for (const keyword of keywords) {
                if (text.includes(keyword)) {
                    return location;
                }
            }
        }
        
        return 'Bangalore'; // Default location
    }
    
    /**
     * Extract education level from resume text with enhanced detection
     * 
     * @param {string} resumeText - Resume text
     * @returns {number} Education level (0-4)
     */
    extractEducation(resumeText) {
        const text = resumeText.toLowerCase();
        
        // PhD level (4)
        if (text.includes('phd') || text.includes('ph.d') || text.includes('doctorate') || 
            text.includes('doctoral')) {
            return 4;
        }
        
        // Master's level (3)
        if (text.includes('master') || text.includes('mtech') || text.includes('mba') || 
            text.includes('m.tech') || text.includes('m.s') || text.includes('ms ') ||
            text.includes('m.sc') || text.includes('mca') || text.includes('m.e') ||
            text.includes('post graduate') || text.includes('postgraduate')) {
            return 3;
        }
        
        // Bachelor's level (2)
        if (text.includes('bachelor') || text.includes('btech') || text.includes('b.tech') ||
            text.includes('be ') || text.includes('b.e') || text.includes('bca') ||
            text.includes('bsc') || text.includes('b.sc') || text.includes('bcom') ||
            text.includes('ba ') || text.includes('b.a') || text.includes('graduate')) {
            return 2;
        }
        
        // Diploma level (1)
        if (text.includes('diploma') || text.includes('polytechnic') || text.includes('iti')) {
            return 1;
        }
        
        // High school or below (0)
        return 0;
    }
    
    /**
     * Estimate market rate based on role, experience, and location for HR system
     * Enhanced with industry-standard salary bands and experience-based progression
     * 
     * @param {string} role - Job role
     * @param {number} experience - Years of experience
     * @param {string} location - Location
     * @returns {number} Estimated market rate in lakhs
     */
    estimateMarketRate(role, experience, location) {
        const experienceGroup = this.categorizeExperience(experience);
        
        // Enhanced salary bands by role and experience group (in lakhs)
        const salaryBands = {
            'Backend Developer': { freshers: 7, junior: 12, mid_level: 18, senior: 28 },
            'Data Scientist': { freshers: 9, junior: 16, mid_level: 25, senior: 40 },
            'DevOps Engineer': { freshers: 8, junior: 14, mid_level: 22, senior: 35 },
            'Frontend Developer': { freshers: 6, junior: 11, mid_level: 16, senior: 25 },
            'Full-stack Developer': { freshers: 7, junior: 13, mid_level: 20, senior: 30 },
            'Mobile Developer': { freshers: 6, junior: 12, mid_level: 18, senior: 28 },
            'Product Manager': { freshers: 10, junior: 18, mid_level: 28, senior: 45 },
            'QA Engineer': { freshers: 5, junior: 9, mid_level: 14, senior: 22 },
            'Tech Lead': { freshers: 12, junior: 20, mid_level: 30, senior: 50 },
            'UI/UX Designer': { freshers: 5, junior: 10, mid_level: 15, senior: 24 }
        };
        
        // Get base salary from experience-specific band
        const roleBands = salaryBands[role] || salaryBands['Backend Developer'];
        let baseSalary;
        
        switch(experienceGroup) {
            case 'freshers':
                baseSalary = roleBands.freshers;
                break;
            case 'junior':
                baseSalary = roleBands.junior;
                break;
            case 'mid_level':
                baseSalary = roleBands.mid_level;
                break;
            case 'senior':
                baseSalary = roleBands.senior;
                break;
            default:
                baseSalary = roleBands.junior;
        }
        
        // Fine-tune based on exact years of experience within the band
        baseSalary = this.adjustForExactExperience(baseSalary, experience, experienceGroup);
        
        // Apply location-based cost of living adjustments
        const locationMultipliers = {
            'Mumbai': 1.4,      // Highest cost of living
            'Bangalore': 1.25,  // Tech hub premium
            'Delhi': 1.15,      // Capital premium
            'Gurgaon': 1.2,     // Corporate hub
            'Noida': 1.15,      // NCR premium
            'Pune': 1.0,        // Baseline
            'Hyderabad': 1.05,  // Growing tech center
            'Chennai': 0.9      // Lower cost of living
        };
        
        const locationMultiplier = locationMultipliers[location] || 1.0;
        baseSalary *= locationMultiplier;
        
        return Math.round(baseSalary * 10) / 10; // Round to 1 decimal place
    }
    
    /**
     * Adjust salary based on exact years of experience within experience group
     * 
     * @param {number} baseSalary - Base salary for experience group
     * @param {number} experience - Years of experience
     * @param {string} experienceGroup - Experience group category
     * @returns {number} Adjusted salary
     */
    adjustForExactExperience(baseSalary, experience, experienceGroup) {
        switch(experienceGroup) {
            case 'freshers':
                // 0-1 years: minimal variation
                return baseSalary + (experience * 0.5);
                
            case 'junior':
                // 1-3 years: steady progression
                const juniorYears = Math.max(0, experience - 1);
                return baseSalary + (juniorYears * 1.5);
                
            case 'mid_level':
                // 3-7 years: significant growth
                const midYears = Math.max(0, experience - 3);
                return baseSalary + (midYears * 2.0);
                
            case 'senior':
                // 7+ years: leadership premium
                const seniorYears = Math.max(0, experience - 7);
                return baseSalary + (Math.min(seniorYears, 8) * 2.5);
                
            default:
                return baseSalary;
        }
    }
    
    /**
     * Estimate performance rating based on ATS scores, achievements, and experience level
     * Enhanced to better assess candidates across all experience groups
     * 
     * @param {Object} resumeAnalysis - ATS analysis results
     * @param {string} resumeText - Resume text
     * @returns {number} Performance rating (1-5)
     */
    estimatePerformance(resumeAnalysis, resumeText) {
        const experience = this.extractExperience(resumeText);
        const experienceGroup = this.categorizeExperience(experience);
        let rating = 3.0; // Base rating
        
        // Factor in ATS scores with experience-adjusted expectations
        if (resumeAnalysis && resumeAnalysis.overallScore) {
            const atsScore = resumeAnalysis.overallScore;
            
            // Adjust ATS score expectations based on experience level
            const atsExpectations = {
                'freshers': 65,    // Lower expectations for freshers
                'junior': 70,      // Standard expectations
                'mid_level': 75,   // Higher expectations
                'senior': 80       // Highest expectations
            };
            
            const expectedScore = atsExpectations[experienceGroup] || 70;
            rating += (atsScore - expectedScore) / 25; // Normalized adjustment
        }
        
        // Experience-specific performance indicators
        rating += this.assessExperienceSpecificPerformance(resumeText, experienceGroup);
        
        // Look for quantifiable achievements
        rating += this.assessQuantifiableAchievements(resumeText, experienceGroup);
        
        // Assess leadership and impact indicators
        rating += this.assessLeadershipImpact(resumeText, experienceGroup);
        
        return Math.max(1.0, Math.min(5.0, Math.round(rating * 10) / 10));
    }
    
    /**
     * Assess performance indicators specific to experience level
     * 
     * @param {string} resumeText - Resume text
     * @param {string} experienceGroup - Experience group
     * @returns {number} Performance adjustment (-1 to +1)
     */
    assessExperienceSpecificPerformance(resumeText, experienceGroup) {
        const text = resumeText.toLowerCase();
        let adjustment = 0;
        
        switch(experienceGroup) {
            case 'freshers':
                // For freshers, focus on learning, projects, and potential
                const fresherIndicators = [
                    'internship', 'project', 'coursework', 'hackathon',
                    'competition', 'certification', 'learning', 'training'
                ];
                
                for (const indicator of fresherIndicators) {
                    if (text.includes(indicator)) {
                        adjustment += 0.1;
                    }
                }
                break;
                
            case 'junior':
                // For junior, focus on skill development and contribution
                const juniorIndicators = [
                    'contributed', 'developed', 'implemented', 'collaborated',
                    'learned', 'adapted', 'supported'
                ];
                
                for (const indicator of juniorIndicators) {
                    if (text.includes(indicator)) {
                        adjustment += 0.08;
                    }
                }
                break;
                
            case 'mid_level':
                // For mid-level, focus on ownership and impact
                const midIndicators = [
                    'owned', 'delivered', 'improved', 'optimized',
                    'designed', 'architected', 'mentored'
                ];
                
                for (const indicator of midIndicators) {
                    if (text.includes(indicator)) {
                        adjustment += 0.1;
                    }
                }
                break;
                
            case 'senior':
                // For senior, focus on leadership and strategic impact
                const seniorIndicators = [
                    'led', 'managed', 'strategic', 'vision', 'transformed',
                    'scaled', 'established', 'pioneered'
                ];
                
                for (const indicator of seniorIndicators) {
                    if (text.includes(indicator)) {
                        adjustment += 0.12;
                    }
                }
                break;
        }
        
        return Math.min(adjustment, 1.0);
    }
    
    /**
     * Assess quantifiable achievements in resume
     * 
     * @param {string} resumeText - Resume text
     * @param {string} experienceGroup - Experience group
     * @returns {number} Achievement score adjustment (0 to +0.8)
     */
    assessQuantifiableAchievements(resumeText, experienceGroup) {
        const text = resumeText.toLowerCase();
        let score = 0;
        
        // Look for percentage improvements
        const percentageMatches = text.match(/\d+%/g) || [];
        score += Math.min(percentageMatches.length * 0.1, 0.3);
        
        // Look for numerical achievements
        const numberMatches = text.match(/\b\d+[kmb]?\b/g) || [];
        score += Math.min(numberMatches.length * 0.05, 0.2);
        
        // Look for time-based achievements
        const timeKeywords = ['reduced time', 'faster', 'efficiency', 'performance'];
        for (const keyword of timeKeywords) {
            if (text.includes(keyword)) {
                score += 0.1;
            }
        }
        
        // Adjust expectations based on experience
        const experienceMultiplier = {
            'freshers': 0.5,    // Lower expectations
            'junior': 0.7,      // Moderate expectations
            'mid_level': 1.0,   // Full expectations
            'senior': 1.2       // Higher expectations
        };
        
        return Math.min(score * (experienceMultiplier[experienceGroup] || 1.0), 0.8);
    }
    
    /**
     * Assess leadership and impact indicators
     * 
     * @param {string} resumeText - Resume text
     * @param {string} experienceGroup - Experience group
     * @returns {number} Leadership score adjustment (0 to +0.5)
     */
    assessLeadershipImpact(resumeText, experienceGroup) {
        const text = resumeText.toLowerCase();
        let score = 0;
        
        const leadershipKeywords = [
            'team lead', 'project lead', 'managed team', 'supervised',
            'coordinated', 'guided', 'mentored', 'trained'
        ];
        
        for (const keyword of leadershipKeywords) {
            if (text.includes(keyword)) {
                score += 0.1;
            }
        }
        
        // Experience-based leadership expectations
        const leadershipExpectations = {
            'freshers': 0.2,    // Minimal leadership expected
            'junior': 0.3,      // Some leadership potential
            'mid_level': 0.4,   // Leadership development
            'senior': 0.5       // Strong leadership expected
        };
        
        return Math.min(score, leadershipExpectations[experienceGroup] || 0.3);
    }
    
    /**
     * Extract leadership score from resume text
     * 
     * @param {string} resumeText - Resume text
     * @returns {number} Leadership score (1-10)
     */
    extractLeadershipScore(resumeText) {
        const text = resumeText.toLowerCase();
        let score = 5; // Base score
        
        const leadershipKeywords = [
            'led', 'managed', 'supervised', 'coordinated', 'mentored',
            'team lead', 'project manager', 'scrum master', 'tech lead'
        ];
        
        for (const keyword of leadershipKeywords) {
            if (text.includes(keyword)) {
                score += 0.5;
            }
        }
        
        // Look for team size mentions
        const teamSizeMatch = text.match(/(\d+)\s*member\s*team/);
        if (teamSizeMatch) {
            const teamSize = parseInt(teamSizeMatch[1]);
            score += Math.min(teamSize * 0.2, 2.0);
        }
        
        return Math.max(1, Math.min(10, Math.round(score * 10) / 10));
    }
    
    /**
     * Extract cross-functional experience indicators
     * 
     * @param {string} resumeText - Resume text
     * @returns {number} Cross-functional experience level
     */
    extractCrossFunctionalExperience(resumeText) {
        const text = resumeText.toLowerCase();
        let score = 0;
        
        const crossFunctionalKeywords = [
            'cross-functional', 'cross functional', 'collaborated',
            'worked with', 'stakeholder', 'business', 'product',
            'design', 'marketing', 'sales'
        ];
        
        for (const keyword of crossFunctionalKeywords) {
            if (text.includes(keyword)) {
                score += 0.5;
            }
        }
        
        return Math.max(0, Math.min(5, Math.round(score)));
    }
    
    /**
     * Assess technical skills based on resume content
     * 
     * @param {string} resumeText - Resume text
     * @returns {number} Technical skills score (1-10)
     */
    assessTechnicalSkills(resumeText) {
        const text = resumeText.toLowerCase();
        let score = 5; // Base score
        
        const technicalKeywords = [
            'programming', 'coding', 'development', 'software', 'algorithm',
            'database', 'api', 'framework', 'library', 'architecture',
            'cloud', 'aws', 'azure', 'docker', 'kubernetes'
        ];
        
        for (const keyword of technicalKeywords) {
            if (text.includes(keyword)) {
                score += 0.3;
            }
        }
        
        return Math.max(1, Math.min(10, Math.round(score * 10) / 10));
    }
    
    /**
     * Assess project complexity based on resume content
     * 
     * @param {string} resumeText - Resume text
     * @returns {number} Project complexity score (1-10)
     */
    assessProjectComplexity(resumeText) {
        const text = resumeText.toLowerCase();
        let score = 5; // Base score
        
        const complexityKeywords = [
            'large scale', 'enterprise', 'microservices', 'distributed',
            'high performance', 'scalable', 'optimization', 'architecture'
        ];
        
        for (const keyword of complexityKeywords) {
            if (text.includes(keyword)) {
                score += 0.5;
            }
        }
        
        return Math.max(1, Math.min(10, Math.round(score * 10) / 10));
    }
    
    /**
     * Assess certifications and additional qualifications
     * 
     * @param {string} resumeText - Resume text
     * @returns {number} Certification score (0-5)
     */
    assessCertifications(resumeText) {
        const text = resumeText.toLowerCase();
        let score = 0;
        
        const certificationKeywords = [
            'certified', 'certification', 'aws certified', 'azure certified',
            'google cloud', 'oracle certified', 'microsoft certified',
            'scrum master', 'pmp', 'cissp'
        ];
        
        for (const keyword of certificationKeywords) {
            if (text.includes(keyword)) {
                score += 0.5;
            }
        }
        
        return Math.max(0, Math.min(5, Math.round(score * 10) / 10));
    }
    
    /**
     * Adjust features based on experience group for better accuracy
     * 
     * @param {Object} features - Extracted features
     * @param {string} resumeText - Resume text
     * @returns {Object} Adjusted features
     */
    adjustFeaturesForExperienceGroup(features, resumeText) {
        const experienceGroup = features.experience_group;
        
        // For freshers, give more weight to education and skills
        if (experienceGroup === 'freshers') {
            // Boost performance rating based on education level
            if (features.education_level >= 3) {
                features.performance_rating += 0.3;
            }
            
            // Boost based on technical skills for freshers
            if (features.technical_skills_score > 7) {
                features.performance_rating += 0.2;
            }
        }
        
        // For senior roles, emphasize leadership and project complexity
        if (experienceGroup === 'senior') {
            if (features.project_complexity_score > 7) {
                features.performance_rating += 0.2;
            }
        }
        
        // Ensure performance rating stays within bounds
        features.performance_rating = Math.max(1.0, Math.min(5.0, features.performance_rating));
        
        return features;
    }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { SalaryPredictor, ResumeFeatureExtractor };
}

// Global instance for browser usage
if (typeof window !== 'undefined') {
    window.SalaryPredictor = SalaryPredictor;
    window.ResumeFeatureExtractor = ResumeFeatureExtractor;
}