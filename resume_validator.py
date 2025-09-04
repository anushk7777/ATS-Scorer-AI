#!/usr/bin/env python3
"""
Resume Content Validator Module

This module provides comprehensive validation and scoring for resume content to:
1. Differentiate between actual resumes and irrelevant PDFs
2. Score resume quality and completeness
3. Detect non-resume documents (invoices, reports, etc.)
4. Validate resume structure and content

Author: OTS Solutions
Version: 1.0.0
Date: 2025-01-25
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Enumeration for different document types"""
    RESUME = "resume"
    CV = "cv"
    INVOICE = "invoice"
    REPORT = "report"
    MANUAL = "manual"
    ACADEMIC_PAPER = "academic_paper"
    LEGAL_DOCUMENT = "legal_document"
    FINANCIAL_STATEMENT = "financial_statement"
    UNKNOWN = "unknown"
    IRRELEVANT = "irrelevant"

class ValidationResult(Enum):
    """Enumeration for validation results"""
    VALID_RESUME = "valid_resume"
    POOR_QUALITY_RESUME = "poor_quality_resume"
    NON_RESUME_DOCUMENT = "non_resume_document"
    INSUFFICIENT_CONTENT = "insufficient_content"
    INVALID_DOCUMENT = "invalid_document"

@dataclass
class ResumeValidationScore:
    """
    Data class to store resume validation results and scoring
    """
    is_valid_resume: bool
    confidence_score: float  # 0.0 to 1.0
    quality_score: float     # 0.0 to 1.0
    document_type: DocumentType
    validation_result: ValidationResult
    content_analysis: Dict[str, Any]
    recommendations: List[str]
    detected_sections: List[str]
    missing_sections: List[str]
    word_count: int
    
class ResumeContentValidator:
    """
    Advanced resume content validator that analyzes document structure,
    content patterns, and quality to determine if a document is a valid resume.
    """
    
    def __init__(self):
        """
        Initialize the resume validator with predefined patterns and rules.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Resume-specific keywords and patterns
        self.resume_indicators = {
            'contact_info': [
                r'\b(?:email|phone|mobile|address|linkedin|github)\b',
                r'\b\d{10}\b',  # Phone numbers
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
                r'\blinkedin\.com/in/\b',
                r'\bgithub\.com/\b'
            ],
            'education_keywords': [
                r'\b(?:education|academic|qualification|degree|diploma|certificate)\b',
                r'\b(?:bachelor|master|phd|b\.?tech|m\.?tech|b\.?e|m\.?e|bca|mca|bba|mba)\b',
                r'\b(?:university|college|institute|school)\b',
                r'\b(?:graduated|graduation|cgpa|gpa|percentage|marks)\b',
                r'\b(?:iit|nit|iiit|bits|vit|srm|amity)\b'
            ],
            'experience_keywords': [
                r'\b(?:experience|work|employment|career|professional)\b',
                r'\b(?:intern|internship|trainee|associate|analyst|developer|engineer|manager)\b',
                r'\b(?:company|organization|firm|corporation|startup)\b',
                r'\b(?:project|responsibility|achievement|accomplishment)\b',
                r'\b(?:years?|months?)\s+(?:of\s+)?(?:experience|work)\b'
            ],
            'skills_keywords': [
                r'\b(?:skills|technical|programming|languages|technologies|tools)\b',
                r'\b(?:python|java|javascript|react|node|angular|sql|aws|docker|kubernetes)\b',
                r'\b(?:proficient|experienced|familiar|knowledge|expertise)\b'
            ],
            'section_headers': [
                r'\b(?:summary|objective|profile|about)\b',
                r'\b(?:education|academic|qualification)\b',
                r'\b(?:experience|work|employment|career)\b',
                r'\b(?:skills|technical|competencies)\b',
                r'\b(?:projects|portfolio|achievements)\b',
                r'\b(?:certifications|awards|honors)\b',
                r'\b(?:references|contact|personal)\b'
            ]
        }
        
        # Non-resume document indicators
        self.non_resume_indicators = {
            'invoice_patterns': [
                r'\b(?:invoice|bill|receipt|payment|amount due|total amount)\b',
                r'\b(?:gst|tax|vat|service charge)\b',
                r'\b(?:vendor|supplier|customer|client)\b',
                r'\$\d+\.\d{2}|₹\d+|\d+\.\d{2}\s*(?:usd|inr|eur)'
            ],
            'report_patterns': [
                r'\b(?:report|analysis|findings|conclusion|methodology)\b',
                r'\b(?:executive summary|abstract|introduction|literature review)\b',
                r'\b(?:data|statistics|metrics|kpi|performance)\b',
                r'\b(?:quarterly|annual|monthly|weekly)\s+(?:report|review)\b'
            ],
            'manual_patterns': [
                r'\b(?:manual|guide|instructions|tutorial|documentation)\b',
                r'\b(?:step|procedure|process|workflow|installation)\b',
                r'\b(?:chapter|section|appendix|table of contents)\b'
            ],
            'academic_patterns': [
                r'\b(?:abstract|introduction|methodology|results|discussion|conclusion)\b',
                r'\b(?:research|study|experiment|hypothesis|literature)\b',
                r'\b(?:references|bibliography|citations|doi)\b',
                r'\b(?:journal|conference|proceedings|publication)\b'
            ],
            'legal_patterns': [
                r'\b(?:contract|agreement|terms|conditions|clause)\b',
                r'\b(?:whereas|therefore|hereby|party|parties)\b',
                r'\b(?:legal|law|court|jurisdiction|liability)\b'
            ],
            'financial_patterns': [
                r'\b(?:balance sheet|income statement|cash flow|financial)\b',
                r'\b(?:assets|liabilities|equity|revenue|expenses)\b',
                r'\b(?:profit|loss|margin|roi|ebitda)\b'
            ]
        }
        
        # Minimum requirements for a valid resume
        self.minimum_requirements = {
            'min_word_count': 50,
            'required_sections': ['contact_info', 'education_keywords'],  # At least these two
            'min_confidence_score': 0.6
        }
    
    def validate_resume(self, resume_text: str) -> ResumeValidationScore:
        """
        Comprehensive validation of resume content.
        
        Args:
            resume_text: Raw text content of the document
            
        Returns:
            ResumeValidationScore object with detailed analysis
        """
        try:
            self.logger.info("Starting resume validation process")
            
            # Basic text preprocessing
            text_lower = resume_text.lower().strip()
            word_count = len(text_lower.split())
            
            # Check minimum word count
            if word_count < self.minimum_requirements['min_word_count']:
                return ResumeValidationScore(
                    is_valid_resume=False,
                    confidence_score=0.0,
                    quality_score=0.0,
                    document_type=DocumentType.INSUFFICIENT_CONTENT,
                    validation_result=ValidationResult.INSUFFICIENT_CONTENT,
                    content_analysis={'word_count': word_count, 'reason': 'Insufficient content'},
                    recommendations=['Document too short to be a meaningful resume'],
                    detected_sections=[],
                    missing_sections=list(self.resume_indicators.keys()),
                    word_count=word_count
                )
            
            # Analyze resume content
            resume_analysis = self._analyze_resume_content(text_lower)
            
            # Analyze non-resume patterns
            non_resume_analysis = self._analyze_non_resume_patterns(text_lower)
            
            # Calculate scores
            confidence_score = self._calculate_confidence_score(resume_analysis, non_resume_analysis)
            quality_score = self._calculate_quality_score(resume_analysis, word_count)
            
            # Determine document type
            document_type = self._determine_document_type(resume_analysis, non_resume_analysis, confidence_score)
            
            # Determine validation result
            validation_result = self._determine_validation_result(confidence_score, quality_score, document_type)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(resume_analysis, quality_score)
            
            # Identify missing sections
            missing_sections = self._identify_missing_sections(resume_analysis)
            
            # Compile content analysis
            content_analysis = {
                'word_count': word_count,
                'resume_indicators_found': resume_analysis,
                'non_resume_indicators_found': non_resume_analysis,
                'section_coverage': len(resume_analysis['detected_sections']) / len(self.resume_indicators),
                'content_density': word_count / max(len(resume_text), 1)
            }
            
            is_valid_resume = validation_result in [ValidationResult.VALID_RESUME, ValidationResult.POOR_QUALITY_RESUME]
            
            self.logger.info(f"Validation completed: {validation_result.value}, confidence: {confidence_score:.2f}")
            
            return ResumeValidationScore(
                is_valid_resume=is_valid_resume,
                confidence_score=confidence_score,
                quality_score=quality_score,
                document_type=document_type,
                validation_result=validation_result,
                content_analysis=content_analysis,
                recommendations=recommendations,
                detected_sections=resume_analysis['detected_sections'],
                missing_sections=missing_sections,
                word_count=word_count
            )
            
        except Exception as e:
            self.logger.error(f"Error during resume validation: {e}")
            return ResumeValidationScore(
                is_valid_resume=False,
                confidence_score=0.0,
                quality_score=0.0,
                document_type=DocumentType.UNKNOWN,
                validation_result=ValidationResult.INVALID_DOCUMENT,
                content_analysis={'error': str(e)},
                recommendations=['Document could not be processed'],
                detected_sections=[],
                missing_sections=list(self.resume_indicators.keys()),
                word_count=0
            )
    
    def _analyze_resume_content(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for resume-specific content patterns.
        
        Args:
            text: Lowercase text content
            
        Returns:
            Dictionary with analysis results
        """
        analysis = {
            'detected_sections': [],
            'section_scores': {},
            'total_matches': 0
        }
        
        for section, patterns in self.resume_indicators.items():
            matches = 0
            for pattern in patterns:
                matches += len(re.findall(pattern, text, re.IGNORECASE))
            
            if matches > 0:
                analysis['detected_sections'].append(section)
                analysis['section_scores'][section] = matches
                analysis['total_matches'] += matches
        
        return analysis
    
    def _analyze_non_resume_patterns(self, text: str) -> Dict[str, Any]:
        """
        Analyze text for non-resume document patterns.
        
        Args:
            text: Lowercase text content
            
        Returns:
            Dictionary with non-resume pattern analysis
        """
        analysis = {
            'detected_types': [],
            'type_scores': {},
            'total_matches': 0
        }
        
        for doc_type, patterns in self.non_resume_indicators.items():
            matches = 0
            for pattern in patterns:
                matches += len(re.findall(pattern, text, re.IGNORECASE))
            
            if matches > 0:
                analysis['detected_types'].append(doc_type)
                analysis['type_scores'][doc_type] = matches
                analysis['total_matches'] += matches
        
        return analysis
    
    def _calculate_confidence_score(self, resume_analysis: Dict, non_resume_analysis: Dict) -> float:
        """
        Calculate confidence score that this is a resume.
        
        Args:
            resume_analysis: Resume content analysis results
            non_resume_analysis: Non-resume pattern analysis results
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        resume_score = len(resume_analysis['detected_sections']) / len(self.resume_indicators)
        non_resume_penalty = min(non_resume_analysis['total_matches'] * 0.1, 0.5)
        
        # Bonus for having required sections
        required_bonus = 0.0
        for required_section in self.minimum_requirements['required_sections']:
            if required_section in resume_analysis['detected_sections']:
                required_bonus += 0.2
        
        confidence = min(resume_score + required_bonus - non_resume_penalty, 1.0)
        return max(confidence, 0.0)
    
    def _calculate_quality_score(self, resume_analysis: Dict, word_count: int) -> float:
        """
        Calculate quality score of the resume content.
        
        Args:
            resume_analysis: Resume content analysis results
            word_count: Total word count
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Base score from section coverage
        section_score = len(resume_analysis['detected_sections']) / len(self.resume_indicators)
        
        # Word count factor (optimal range: 200-800 words)
        if 200 <= word_count <= 800:
            word_factor = 1.0
        elif word_count < 200:
            word_factor = word_count / 200
        else:
            word_factor = max(0.5, 800 / word_count)
        
        # Content density factor
        content_density = resume_analysis['total_matches'] / max(word_count, 1)
        density_factor = min(content_density * 100, 1.0)  # Normalize
        
        quality = (section_score * 0.5 + word_factor * 0.3 + density_factor * 0.2)
        return min(quality, 1.0)
    
    def _determine_document_type(self, resume_analysis: Dict, non_resume_analysis: Dict, confidence: float) -> DocumentType:
        """
        Determine the most likely document type.
        
        Args:
            resume_analysis: Resume content analysis
            non_resume_analysis: Non-resume pattern analysis
            confidence: Confidence score
            
        Returns:
            DocumentType enum value
        """
        if confidence >= 0.7:
            return DocumentType.RESUME
        elif confidence >= 0.4:
            return DocumentType.CV  # Could be a CV or informal resume
        
        # Check for specific non-resume types
        if non_resume_analysis['detected_types']:
            strongest_type = max(non_resume_analysis['type_scores'], key=non_resume_analysis['type_scores'].get)
            type_mapping = {
                'invoice_patterns': DocumentType.INVOICE,
                'report_patterns': DocumentType.REPORT,
                'manual_patterns': DocumentType.MANUAL,
                'academic_patterns': DocumentType.ACADEMIC_PAPER,
                'legal_patterns': DocumentType.LEGAL_DOCUMENT,
                'financial_patterns': DocumentType.FINANCIAL_STATEMENT
            }
            return type_mapping.get(strongest_type, DocumentType.IRRELEVANT)
        
        return DocumentType.UNKNOWN
    
    def _determine_validation_result(self, confidence: float, quality: float, doc_type: DocumentType) -> ValidationResult:
        """
        Determine the final validation result.
        
        Args:
            confidence: Confidence score
            quality: Quality score
            doc_type: Document type
            
        Returns:
            ValidationResult enum value
        """
        if doc_type in [DocumentType.RESUME, DocumentType.CV]:
            if confidence >= 0.7 and quality >= 0.6:
                return ValidationResult.VALID_RESUME
            elif confidence >= 0.4:
                return ValidationResult.POOR_QUALITY_RESUME
        
        if doc_type in [DocumentType.UNKNOWN]:
            return ValidationResult.INSUFFICIENT_CONTENT
        
        return ValidationResult.NON_RESUME_DOCUMENT
    
    def _generate_recommendations(self, resume_analysis: Dict, quality_score: float) -> List[str]:
        """
        Generate recommendations for improving resume quality.
        
        Args:
            resume_analysis: Resume content analysis
            quality_score: Quality score
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        if 'contact_info' not in resume_analysis['detected_sections']:
            recommendations.append("Add clear contact information (email, phone, address)")
        
        if 'education_keywords' not in resume_analysis['detected_sections']:
            recommendations.append("Include education details (degree, institution, graduation year)")
        
        if 'experience_keywords' not in resume_analysis['detected_sections']:
            recommendations.append("Add work experience or internship details")
        
        if 'skills_keywords' not in resume_analysis['detected_sections']:
            recommendations.append("List relevant technical and soft skills")
        
        if quality_score < 0.5:
            recommendations.append("Expand content with more detailed descriptions")
            recommendations.append("Use action verbs and quantify achievements")
        
        if not recommendations:
            recommendations.append("Resume looks good! Consider adding more specific achievements and metrics")
        
        return recommendations
    
    def _identify_missing_sections(self, resume_analysis: Dict) -> List[str]:
        """
        Identify missing resume sections.
        
        Args:
            resume_analysis: Resume content analysis
            
        Returns:
            List of missing section names
        """
        detected = set(resume_analysis['detected_sections'])
        all_sections = set(self.resume_indicators.keys())
        return list(all_sections - detected)
    
    def is_valid_resume(self, resume_text: str, min_confidence: float = 0.6) -> bool:
        """
        Quick validation check to determine if document is a valid resume.
        
        Args:
            resume_text: Raw text content
            min_confidence: Minimum confidence threshold
            
        Returns:
            Boolean indicating if document is a valid resume
        """
        validation_result = self.validate_resume(resume_text)
        return validation_result.confidence_score >= min_confidence and validation_result.is_valid_resume
    
    def get_validation_summary(self, resume_text: str) -> Dict[str, Any]:
        """
        Get a summary of validation results in a simple format.
        
        Args:
            resume_text: Raw text content
            
        Returns:
            Dictionary with validation summary
        """
        validation = self.validate_resume(resume_text)
        
        return {
            'is_valid_resume': validation.is_valid_resume,
            'confidence_score': round(validation.confidence_score, 2),
            'quality_score': round(validation.quality_score, 2),
            'document_type': validation.document_type.value,
            'validation_result': validation.validation_result.value,
            'word_count': validation.word_count,
            'detected_sections': validation.detected_sections,
            'missing_sections': validation.missing_sections,
            'top_recommendations': validation.recommendations[:3]  # Top 3 recommendations
        }

# Example usage and testing
if __name__ == "__main__":
    # Initialize validator
    validator = ResumeContentValidator()
    
    # Test with sample resume text
    sample_resume = """
    John Doe
    Software Developer
    Email: john.doe@email.com
    Phone: +91-9876543210
    
    Education:
    B.Tech Computer Science, IIT Delhi (2020)
    CGPA: 8.5/10
    
    Experience:
    Software Developer at TechCorp (2020-2023)
    - Developed web applications using React and Node.js
    - Improved system performance by 30%
    
    Skills:
    Python, JavaScript, React, Node.js, SQL, AWS
    
    Projects:
    E-commerce Platform - Built a full-stack application
    """
    
    # Test with non-resume document
    sample_invoice = """
    INVOICE #INV-2023-001
    
    Bill To: ABC Company
    Amount Due: $1,500.00
    GST: 18%
    Total Amount: $1,770.00
    
    Services Provided:
    - Web Development Services
    - Consulting Hours: 50
    
    Payment Terms: Net 30 days
    """
    
    print("=== Resume Validation Test ===")
    
    # Test valid resume
    print("\n1. Testing Valid Resume:")
    result = validator.get_validation_summary(sample_resume)
    print(json.dumps(result, indent=2))
    
    # Test invalid document (invoice)
    print("\n2. Testing Non-Resume Document (Invoice):")
    result = validator.get_validation_summary(sample_invoice)
    print(json.dumps(result, indent=2))
    
    # Test quick validation
    print("\n3. Quick Validation Results:")
    print(f"Resume is valid: {validator.is_valid_resume(sample_resume)}")
    print(f"Invoice is valid resume: {validator.is_valid_resume(sample_invoice)}")
    
    print("\n=== Validation Test Completed ===")