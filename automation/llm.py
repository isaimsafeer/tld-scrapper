import google.generativeai as genai
import json
import logging
from typing import Dict, Any, List, Optional
import openpyxl
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class GeminiProcessor:
    """
    Process scraped table data using Gemini AI to clean and structure it
    """
    
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize Gemini processor
        
        Args:
            api_key: Google AI API key
            model_name: Gemini model to use
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Initialized Gemini processor with model: {model_name}")
    
    def read_excel_data(self, excel_file: str) -> Dict[str, Any]:
        """
        Read data from Excel file
        
        Args:
            excel_file: Path to Excel file
            
        Returns:
            Dictionary containing headers and rows
        """
        try:
            workbook = openpyxl.load_workbook(excel_file)
            sheet = workbook.active
            
            # Get all rows
            rows = list(sheet.iter_rows(values_only=True))
            
            if not rows:
                return {"headers": [], "data": []}
            
            # First row is headers
            headers = [str(h).strip() if h else f"Column_{i}" for i, h in enumerate(rows[0])]
            data_rows = [list(row) for row in rows[1:]]
            
            logger.info(f"Read {len(data_rows)} rows with {len(headers)} columns from {excel_file}")
            
            return {
                "headers": headers,
                "data": data_rows
            }
        except Exception as e:
            logger.error(f"Error reading Excel file: {e}")
            raise
    
    def process_table_data(self, excel_file: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Process table data using Gemini to clean and structure it
        
        Args:
            excel_file: Path to Excel file with scraped data
            context: Optional context about what the table contains
            
        Returns:
            Dictionary containing processed data
        """
        try:
            # Read Excel data
            excel_data = self.read_excel_data(excel_file)
            
            if not excel_data["data"]:
                logger.warning("No data found in Excel file")
                return {"status": "error", "message": "No data found"}
            
            # Sample data if too large (send first 50 rows for analysis)
            sample_data = {
                "headers": excel_data["headers"],
                "data": excel_data["data"][:50]
            }
            
            # Prepare data for Gemini
            table_text = self._format_table_for_llm(sample_data)
            
            # Create prompt for Gemini
            prompt = self._create_processing_prompt(table_text, context, len(excel_data["data"]))
            
            # Call Gemini
            logger.info("Sending data to Gemini for processing...")
            response = self.model.generate_content(prompt)
            
            # Parse response
            processed_data = self._parse_gemini_response(response.text)
            
            # Add metadata
            if processed_data.get("status") == "success":
                processed_data["metadata"] = {
                    "source_file": excel_file,
                    "total_rows": len(excel_data["data"]),
                    "total_columns": len(excel_data["headers"]),
                    "columns": excel_data["headers"]
                }
            
            logger.info("Data processing completed")
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing table data: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _format_table_for_llm(self, excel_data: Dict[str, Any]) -> str:
        """
        Format table data as text for LLM processing (more readable format)
        
        Args:
            excel_data: Dictionary with headers and data
            
        Returns:
            Formatted table text
        """
        lines = []
        headers = excel_data["headers"]
        
        # Create header line
        lines.append("=" * 100)
        lines.append("TABLE HEADERS:")
        lines.append(", ".join(f"[{i+1}] {h}" for i, h in enumerate(headers)))
        lines.append("=" * 100)
        lines.append("")
        
        # Add sample rows in a cleaner format
        lines.append("SAMPLE DATA (First 50 rows):")
        lines.append("")
        
        for idx, row in enumerate(excel_data["data"], 1):
            lines.append(f"--- Row {idx} ---")
            for col_idx, (header, value) in enumerate(zip(headers, row)):
                if value:  # Only show non-empty values
                    lines.append(f"  {header}: {value}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _create_processing_prompt(self, table_text: str, context: Optional[str] = None, total_rows: int = 0) -> str:
        """
        Create enhanced prompt for Gemini to process the table data
        
        Args:
            table_text: Formatted table text
            context: Optional context
            total_rows: Total number of rows in dataset
            
        Returns:
            Processing prompt
        """
        base_context = context or "This is a table of data scraped from a website."
        
        prompt = f"""You are an expert data analyst. Analyze this scraped table data and provide clear, actionable insights.

CONTEXT: {base_context}
TOTAL ROWS IN DATASET: {total_rows}

{table_text}

Please analyze this data and return a well-structured JSON response with the following:

1. **Data Summary**: Overview of what the data contains
2. **Key Findings**: Most important insights from the data
3. **Data Quality Assessment**: Any issues or patterns noticed
4. **Structured Records**: Clean, organized version of the data
5. **Recommendations**: What can be done with this data

Return your response in this EXACT JSON structure:

{{
    "summary": {{
        "description": "Brief description of what this data represents",
        "total_records": <number>,
        "key_columns": ["column1", "column2"],
        "data_types": {{"column": "type"}},
        "date_range": "if applicable",
        "data_quality_score": "Excellent/Good/Fair/Poor"
    }},
    "key_findings": [
        {{
            "finding": "Main insight",
            "importance": "High/Medium/Low",
            "details": "Explanation"
        }}
    ],
    "data_quality": {{
        "completeness": "percentage or description",
        "consistency": "assessment",
        "issues_found": ["issue1", "issue2"],
        "recommendations": ["recommendation1", "recommendation2"]
    }},
    "structured_records": [
        {{
            "record_id": 1,
            "parsed_data": {{
                "field1": "value1",
                "field2": "value2"
            }},
            "highlights": ["notable point about this record"]
        }}
    ],
    "insights": {{
        "top_items": ["item1", "item2"],
        "trends": ["trend1", "trend2"],
        "anomalies": ["anomaly1", "anomaly2"],
        "opportunities": ["opportunity1", "opportunity2"]
    }},
    "recommendations": [
        {{
            "action": "What to do",
            "priority": "High/Medium/Low",
            "expected_outcome": "What you'll achieve"
        }}
    ]
}}

IMPORTANT INSTRUCTIONS:
- Parse ALL numeric values properly (remove currency symbols, convert to numbers)
- Identify and extract any codes, identifiers, or special values
- Group similar items together
- Highlight the most important or extreme values (cheapest, most expensive, etc.)
- If this is pricing data, identify best deals
- Return ONLY valid JSON, no markdown formatting, no extra text
- Be specific and actionable in your insights

JSON Response:"""

        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Gemini's response and extract JSON
        
        Args:
            response_text: Raw response from Gemini
            
        Returns:
            Parsed JSON data
        """
        try:
            # Clean the response
            cleaned_text = response_text.strip()
            
            # Remove markdown code blocks
            if "```json" in cleaned_text:
                cleaned_text = cleaned_text.split("```json")[1].split("```")[0].strip()
            elif "```" in cleaned_text:
                # Try to extract JSON from any code block
                parts = cleaned_text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("{") or part.startswith("["):
                        cleaned_text = part
                        break
            
            # Remove any leading/trailing text that's not JSON
            start_idx = cleaned_text.find("{")
            end_idx = cleaned_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                cleaned_text = cleaned_text[start_idx:end_idx]
            
            # Parse JSON
            data = json.loads(cleaned_text)
            data["status"] = "success"
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Cleaned text: {cleaned_text[:500]}")
            
            # Return a structured error response
            return {
                "status": "error",
                "message": "Failed to parse LLM response as JSON",
                "error_details": str(e),
                "raw_response_preview": response_text[:500] + "..." if len(response_text) > 500 else response_text
            }
    
    def save_processed_data(self, processed_data: Dict[str, Any], output_file: str):
        """
        Save processed data to a nicely formatted JSON file
        
        Args:
            processed_data: Processed data dictionary
            output_file: Output file path
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Processed data saved to {output_file}")
            
            # Also create a human-readable summary file
            summary_file = output_file.replace('.json', '_summary.txt')
            self._create_summary_file(processed_data, summary_file)
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def _create_summary_file(self, processed_data: Dict[str, Any], summary_file: str):
        """
        Create a human-readable summary file
        
        Args:
            processed_data: Processed data dictionary
            summary_file: Output summary file path
        """
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("DATA PROCESSING SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                
                if processed_data.get("status") == "success":
                    # Summary section
                    if "summary" in processed_data:
                        summary = processed_data["summary"]
                        f.write("OVERVIEW\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"Description: {summary.get('description', 'N/A')}\n")
                        f.write(f"Total Records: {summary.get('total_records', 'N/A')}\n")
                        f.write(f"Data Quality: {summary.get('data_quality_score', 'N/A')}\n")
                        f.write("\n")
                    
                    # Key findings
                    if "key_findings" in processed_data:
                        f.write("KEY FINDINGS\n")
                        f.write("-" * 80 + "\n")
                        for idx, finding in enumerate(processed_data["key_findings"], 1):
                            f.write(f"{idx}. [{finding.get('importance', 'N/A')}] {finding.get('finding', 'N/A')}\n")
                            f.write(f"   {finding.get('details', '')}\n\n")
                    
                    # Insights
                    if "insights" in processed_data:
                        insights = processed_data["insights"]
                        f.write("INSIGHTS\n")
                        f.write("-" * 80 + "\n")
                        
                        if insights.get("top_items"):
                            f.write("\nTop Items:\n")
                            for item in insights["top_items"]:
                                f.write(f"  • {item}\n")
                        
                        if insights.get("trends"):
                            f.write("\nTrends:\n")
                            for trend in insights["trends"]:
                                f.write(f"  • {trend}\n")
                        
                        if insights.get("opportunities"):
                            f.write("\nOpportunities:\n")
                            for opp in insights["opportunities"]:
                                f.write(f"  • {opp}\n")
                        f.write("\n")
                    
                    # Recommendations
                    if "recommendations" in processed_data:
                        f.write("RECOMMENDATIONS\n")
                        f.write("-" * 80 + "\n")
                        for idx, rec in enumerate(processed_data["recommendations"], 1):
                            f.write(f"{idx}. [{rec.get('priority', 'N/A')}] {rec.get('action', 'N/A')}\n")
                            f.write(f"   Expected Outcome: {rec.get('expected_outcome', '')}\n\n")
                
                else:
                    f.write("ERROR: Processing failed\n")
                    f.write(f"Message: {processed_data.get('message', 'Unknown error')}\n")
                
                f.write("\n" + "=" * 80 + "\n")
            
            logger.info(f"Summary file created: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Could not create summary file: {e}")


def process_scraped_data(excel_file: str, api_key: str, output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to process scraped data
    
    Args:
        excel_file: Path to Excel file with scraped data
        api_key: Google AI API key
        output_file: Optional output file for processed data
        
    Returns:
        Processed data dictionary
    """
    processor = GeminiProcessor(api_key)
    processed_data = processor.process_table_data(excel_file)
    
    if output_file and processed_data.get("status") == "success":
        processor.save_processed_data(processed_data, output_file)
    
    return processed_data


# Example usage
if __name__ == "__main__":
    import os
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Get API key from environment
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    
    if not api_key:
        print("ERROR: GOOGLE_AI_API_KEY environment variable not set")
        exit(1)
    
    # Example: Process a scraped Excel file
    excel_file = "table_data_example.xlsx"
    output_file = "processed_data.json"
    
    if Path(excel_file).exists():
        result = process_scraped_data(excel_file, api_key, output_file)
        
        if result.get("status") == "success":
            print("\n" + "=" * 80)
            print("PROCESSING SUCCESSFUL")
            print("=" * 80)
            
            if "summary" in result:
                summary = result["summary"]
                print(f"\nDescription: {summary.get('description')}")
                print(f"Total Records: {summary.get('total_records')}")
                print(f"Data Quality: {summary.get('data_quality_score')}")
            
            if "key_findings" in result:
                print("\nKey Findings:")
                for finding in result["key_findings"][:3]:  # Show top 3
                    print(f"  • [{finding.get('importance')}] {finding.get('finding')}")
            
            if "recommendations" in result:
                print("\nTop Recommendations:")
                for rec in result["recommendations"][:3]:  # Show top 3
                    print(f"  • [{rec.get('priority')}] {rec.get('action')}")
            
            print(f"\nFull results saved to: {output_file}")
            print(f"Summary saved to: {output_file.replace('.json', '_summary.txt')}")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("PROCESSING FAILED")
            print("=" * 80)
            print(f"Error: {result.get('message', 'Unknown error')}")
            if result.get('raw_response_preview'):
                print(f"\nResponse preview:\n{result['raw_response_preview']}")
            print("=" * 80)
    else:
        print(f"Excel file not found: {excel_file}")