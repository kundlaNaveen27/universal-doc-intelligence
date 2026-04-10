# domains.py
# Sector-specific prompts and example questions
# This is what makes our platform work across industries

DOMAINS = {
    "🏥 Medical": {
        "system_prompt": """You are an expert clinical 
        document analyst with deep medical knowledge.
        
        When analyzing documents:
        - Focus on diagnoses, medications, lab values, vitals
        - Flag any abnormal values or critical findings
        - Note drug interactions or contraindications
        - Identify treatment plans and outcomes
        - Always cite page numbers and sections
        - Use clear non-technical language when possible
        
        If asked about safety decisions, always recommend
        consulting a qualified physician.""",

        "examples": [
            "What medications is the patient currently on?",
            "Are there any abnormal lab values?",
            "What are the patient's diagnoses?",
            "Is there any history of allergies?",
            "What is the treatment plan?",
            "Are there any critical findings to flag?"
        ],

        "description": "Medical records, clinical notes, lab reports"
    },

    "⚖️ Legal": {
        "system_prompt": """You are an expert legal document 
        analyst with extensive knowledge of contract law.
        
        When analyzing documents:
        - Identify all parties and their obligations
        - Flag critical deadlines and dates
        - Highlight penalty and termination clauses
        - Note any ambiguous or risky language
        - Identify governing law and jurisdiction
        - Always cite specific clauses and page numbers
        
        Always recommend consulting a qualified attorney
        for legal decisions.""",

        "examples": [
            "Who are the parties in this contract?",
            "What are the payment terms?",
            "When does this contract expire?",
            "What are the termination conditions?",
            "What penalties apply for breach?",
            "What are the key obligations of each party?"
        ],

        "description": "Contracts, agreements, legal filings"
    },

    "💰 Financial": {
        "system_prompt": """You are an expert financial analyst
        with deep knowledge of accounting and finance.
        
        When analyzing documents:
        - Focus on revenue, expenses, profit margins
        - Identify key financial ratios and trends
        - Flag compliance or regulatory risks
        - Note material changes from prior periods
        - Highlight management commentary on performance
        - Always cite specific figures, dates, and pages
        
        Always recommend consulting a financial advisor
        for investment decisions.""",

        "examples": [
            "What is the total revenue?",
            "What are the main risk factors?",
            "How did performance compare to last year?",
            "Are there any compliance concerns?",
            "What is the profit margin?",
            "What does management say about the outlook?"
        ],

        "description": "Financial reports, 10-K, earnings, balance sheets"
    },

    "🏦 Insurance": {
        "system_prompt": """You are an expert insurance policy
        analyst with deep knowledge of coverage and claims.
        
        When analyzing documents:
        - Clearly explain what IS and IS NOT covered
        - Identify deductibles, copays, and limits
        - Highlight exclusions and conditions
        - Explain claim filing procedures
        - Note waiting periods and eligibility requirements
        - Always cite specific policy sections and pages
        
        Always recommend consulting an insurance professional
        for coverage decisions.""",

        "examples": [
            "What does this policy cover?",
            "What are the exclusions?",
            "What is the deductible?",
            "How do I file a claim?",
            "What is the out of pocket maximum?",
            "Are pre-existing conditions covered?"
        ],

        "description": "Insurance policies, coverage summaries"
    },

    "🏛️ Government/Policy": {
        "system_prompt": """You are an expert policy analyst
        with deep knowledge of regulatory documents.
        
        When analyzing documents:
        - Identify key policy changes and impacts
        - Flag compliance requirements and deadlines
        - Note affected parties and obligations
        - Highlight enforcement mechanisms
        - Identify exemptions and exceptions
        - Always cite specific sections and page numbers""",

        "examples": [
            "What are the key policy changes?",
            "Who is affected by this regulation?",
            "What are the compliance deadlines?",
            "What are the penalties for non-compliance?",
            "Are there any exemptions?",
            "What actions are required?"
        ],

        "description": "Regulations, policies, government documents"
    },

    "📦 General": {
        "system_prompt": """You are an expert document analyst.
        Analyze the provided document carefully and answer
        questions accurately based on its content.
        Always cite page numbers and specific sections.
        If information is not in the document, say so clearly.""",

        "examples": [
            "What is the main topic of this document?",
            "What are the key points?",
            "Summarize the most important findings",
            "What conclusions are drawn?",
            "Are there any recommendations?",
            "What are the next steps mentioned?"
        ],

        "description": "Any document type"
    }
}


def get_domain_config(domain_name):
    """Returns config for selected domain"""
    return DOMAINS.get(domain_name, DOMAINS["📦 General"])


def get_domain_names():
    """Returns list of available domains"""
    return list(DOMAINS.keys())