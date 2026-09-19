import pandas as pd
import re
import os
from collections import defaultdict
import logging
import sys
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def safe_string_conversion(value):
    """Safely convert any value to a string, handling NaN and None values."""
    if pd.isna(value):
        return ""
    else:
        return str(value).strip()


def validate_dataframe(df, required_columns):
    """
    Validate that the dataframe has the required columns.
    Returns True if valid, False otherwise.
    """
    for column in required_columns:
        if column not in df.columns:
            return False
    return True


def convert_triples_to_qa_pairs(input_xlsx, output_xlsx, error_log=None):
    """
    Convert knowledge graph triples into question-answer pairs,
    consolidating related information where appropriate.

    Args:
        input_xlsx: Path to the input Excel file containing the filtered triples
        output_xlsx: Path to save the output Excel file with QA pairs
        error_log: Optional path to save error log

    Returns:
        DataFrame of generated QA pairs or None if processing failed
    """
    try:
        logger.info(f"Starting conversion from {input_xlsx} to {output_xlsx}")

        # Set up error logging
        if error_log:
            file_handler = logging.FileHandler(error_log)
            file_handler.setLevel(logging.WARNING)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Step 1: Load triples data
        triples_df = load_triples_data(input_xlsx)
        if triples_df is None or len(triples_df) == 0:
            logger.error("Failed to load valid triples data")
            return None

        # Step 2: Prepare and clean the data
        triples_df = prepare_triples_data(triples_df)

        # Step 3: Group triples by entity and relation
        entity_relation_groups, reverse_entity_relation_groups = group_triples(triples_df)

        # Step 4: Generate QA pairs
        qa_df = generate_qa_pairs(entity_relation_groups, reverse_entity_relation_groups)

        # Step 5: Save results
        save_qa_pairs(qa_df, output_xlsx)

        logger.info(f"Conversion completed successfully with {len(qa_df)} QA pairs generated")
        return qa_df

    except Exception as e:
        logger.error(f"Unexpected error during conversion: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def load_triples_data(input_xlsx):
    """
    Load triples data from Excel file, trying various sheet names and formats.
    Returns a DataFrame of triples or None if loading fails.
    """
    logger.info(f"Loading triples from {input_xlsx}...")

    # Define potential sheet names to try
    sheet_names_to_try = [
        'All_Filtered_Triples',  # Main sheet in filtered output
        None,  # Default first sheet
        'Sheet1',  # Common default sheet name
        'Triples'  # Another possible sheet name
    ]

    # Also try layer-specific sheets
    layer_sheets = [f'Layer{i}_Direct' for i in range(1, 5)] + [f'Layer{i}' for i in range(1, 5)]
    sheet_names_to_try.extend(layer_sheets)

    # Try each sheet name
    triples_df = None
    successful_sheet = None

    for sheet_name in sheet_names_to_try:
        try:
            if sheet_name is None:
                logger.info("Trying to load from the first sheet...")
            else:
                logger.info(f"Trying to load from sheet: {sheet_name}")

            df = pd.read_excel(input_xlsx, sheet_name=sheet_name)

            if len(df) > 0:
                logger.info(
                    f"Successfully loaded {len(df)} rows from {'first sheet' if sheet_name is None else sheet_name}")

                # Check if this looks like a triples dataframe (has at least 3 columns)
                if len(df.columns) >= 3:
                    triples_df = df
                    successful_sheet = sheet_name if sheet_name is not None else "first sheet"
                    break
                else:
                    logger.warning(f"Sheet has fewer than 3 columns, may not be triples data")
            else:
                logger.warning(f"Sheet is empty")

        except Exception as e:
            logger.warning(f"Could not load from {'first sheet' if sheet_name is None else sheet_name}: {str(e)}")

    # If we couldn't load from a single sheet, try combining layer sheets
    if triples_df is None:
        try:
            logger.info("Attempting to combine data from layer sheets...")
            layer_dfs = []

            for layer_sheet in layer_sheets:
                try:
                    layer_df = pd.read_excel(input_xlsx, sheet_name=layer_sheet)
                    if len(layer_df) > 0 and len(layer_df.columns) >= 3:
                        layer_dfs.append(layer_df)
                        logger.info(f"Added {len(layer_df)} rows from {layer_sheet}")
                except:
                    pass

            if layer_dfs:
                triples_df = pd.concat(layer_dfs, ignore_index=True).drop_duplicates()
                logger.info(f"Combined {len(triples_df)} unique triples from {len(layer_dfs)} layer sheets")
                successful_sheet = "combined layers"
        except Exception as e:
            logger.warning(f"Failed to combine layer sheets: {str(e)}")

    # Check if we successfully loaded data
    if triples_df is None or len(triples_df) == 0:
        logger.error("Could not load triples data from any sheet")
        return None

    logger.info(f"Successfully loaded {len(triples_df)} triples from {successful_sheet}")
    return triples_df


def prepare_triples_data(df):
    """
    Prepare and clean the triples dataframe by:
    1. Ensuring column names are correct
    2. Converting values to strings
    3. Removing duplicates
    4. Handling missing values

    Returns a cleaned DataFrame.
    """
    logger.info("Preparing and cleaning triples data...")

    # Check column names and rename if necessary
    logger.info(f"Column names in the data: {df.columns.tolist()}")

    # Make sure we have the expected column names
    if not validate_dataframe(df, ['head', 'relation', 'tail']):
        logger.warning("Expected column names not found, attempting to rename columns")

        # Try to use the first three columns as head, relation, tail
        if len(df.columns) >= 3:
            column_mapping = {
                df.columns[0]: 'head',
                df.columns[1]: 'relation',
                df.columns[2]: 'tail'
            }
            df = df.rename(columns=column_mapping)
            logger.info(f"Renamed columns to: {df.columns.tolist()}")
        else:
            raise ValueError("Data does not have enough columns for head, relation, tail")

    # Validate that the dataframe now has the required columns
    if not validate_dataframe(df, ['head', 'relation', 'tail']):
        raise ValueError("Could not establish required head, relation, tail columns")

    # Convert columns to string and handle missing values
    df['head'] = df['head'].apply(safe_string_conversion)
    df['relation'] = df['relation'].apply(safe_string_conversion)
    df['tail'] = df['tail'].apply(safe_string_conversion)

    # Remove rows with empty values in essential columns
    initial_count = len(df)
    df = df[df['head'].str.strip() != ""]
    df = df[df['relation'].str.strip() != ""]
    df = df[df['tail'].str.strip() != ""]
    removed_count = initial_count - len(df)
    if removed_count > 0:
        logger.warning(f"Removed {removed_count} rows with empty values")

    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    removed_count = initial_count - len(df)
    if removed_count > 0:
        logger.info(f"Removed {removed_count} duplicate triples")

    logger.info(f"Data preparation complete. Final dataset has {len(df)} unique triples")
    return df


def group_triples(triples_df):
    """
    Group triples by entity and relation type for efficient QA generation.

    Returns two dictionaries:
    1. entity_relation_groups: Entity -> Relation -> List of related items
    2. reverse_entity_relation_groups: Entity -> Reverse Relation -> List of related items
    """
    logger.info("Grouping triples by entity and relation...")

    # Define which relations should have reverse questions created
    reverse_relations = [
        'disease_symptom',
        'disease_bacteria',
        'disease_complication',
        'antibiotic_bacteria',
        'disease_treatment'
    ]

    # Group related triples by entity and relation type
    entity_relation_groups = defaultdict(lambda: defaultdict(list))

    # Process all triples and group them by head entity and relation
    for _, row in triples_df.iterrows():
        head = row['head']
        relation = row['relation']
        tail = row['tail']

        # Group by head entity and relation
        entity_relation_groups[head][relation].append(tail)

    # Group by tail entity for reverse questions
    reverse_entity_relation_groups = defaultdict(lambda: defaultdict(list))
    for _, row in triples_df.iterrows():
        head = row['head']
        relation = row['relation']
        tail = row['tail']

        # Only create reverse mappings for certain relation types
        if relation in reverse_relations:
            # Group by tail entity and relation (for reverse questions)
            reverse_entity_relation_groups[tail][f"reverse_{relation}"].append(head)

    # Log statistics
    entity_count = len(entity_relation_groups)
    relation_types = set()
    for entity_dict in entity_relation_groups.values():
        relation_types.update(entity_dict.keys())

    reverse_entity_count = len(reverse_entity_relation_groups)
    reverse_relation_types = set()
    for entity_dict in reverse_entity_relation_groups.values():
        reverse_relation_types.update(entity_dict.keys())

    logger.info(f"Grouped {entity_count} entities with {len(relation_types)} relation types")
    logger.info(
        f"Created reverse mappings for {reverse_entity_count} entities with {len(reverse_relation_types)} reverse relation types")

    return entity_relation_groups, reverse_entity_relation_groups


def generate_qa_pairs(entity_relation_groups, reverse_entity_relation_groups):
    """
    Generate question-answer pairs from grouped triples.

    Returns a DataFrame of QA pairs.
    """
    logger.info("Generating question-answer pairs...")

    # Define question templates for different relation types
    question_templates = {
        # Direct question templates
        'disease_symptom': "What are the symptoms of {entity}?",
        'disease_bacteria': "What bacteria are associated with {entity}?",
        'disease_complication': "What are the complications of {entity}?",
        'disease_treatment': "What treatments are available for {entity}?",
        'antibiotic_bacteria': "Which bacteria is {entity} effective against?",
        'bacteria_infection_site': "Which part of the body does {entity} typically infect?",
        'drug_target': "What is the target of the drug {entity}?",
        'type': "What type of entity is {entity}?",
        'description': "Can you describe {entity}?",
        'contraindication_drug': "What drugs are contraindicated with {entity}?",
        'contraindication_situation': "In what situations is {entity} contraindicated?",

        # Reverse question templates
        'reverse_disease_symptom': "Which diseases have {entity} as a symptom?",
        'reverse_disease_bacteria': "Which diseases are associated with {entity}?",
        'reverse_disease_complication': "Which diseases can lead to {entity} as a complication?",
        'reverse_antibiotic_bacteria': "Which antibiotics are effective against {entity}?",
        'reverse_disease_treatment': "Which diseases can be treated with {entity}?",
    }

    # Generate QA pairs
    qa_pairs = []

    # Process direct questions (entity -> related items)
    logger.info("Generating direct question-answer pairs...")
    for entity, relation_dict in entity_relation_groups.items():
        for relation, tail_values in relation_dict.items():
            if relation in question_templates:
                # Create question
                question = question_templates[relation].format(entity=entity)

                # Create answer - join all related values with commas
                answer = ", ".join(sorted(set(tail_values)))

                # Add the QA pair
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'entity': entity,
                    'relation': relation
                })

    # Process reverse questions (entity <- related items)
    logger.info("Generating reverse question-answer pairs...")
    for entity, relation_dict in reverse_entity_relation_groups.items():
        for relation, head_values in relation_dict.items():
            if relation in question_templates:
                # Create question
                question = question_templates[relation].format(entity=entity)

                # Create answer - join all related values with commas
                answer = ", ".join(sorted(set(head_values)))

                # Add the QA pair
                qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'entity': entity,
                    'relation': relation
                })

    # Create dataframe from QA pairs
    qa_df = pd.DataFrame(qa_pairs)
    logger.info(f"Generated {len(qa_df)} basic question-answer pairs")

    # Generate more complex QA pairs by combining related information
    logger.info("Generating complex question-answer pairs...")
    complex_qa_pairs = []

    # Tracking for complex question generation
    complex_questions_generated = 0

    # Example: "What are the symptoms and complications of Disease X?"
    for entity in entity_relation_groups.keys():
        # Check if entity has both symptoms and complications
        if 'disease_symptom' in entity_relation_groups[entity] and 'disease_complication' in entity_relation_groups[
            entity]:
            symptoms = sorted(set(entity_relation_groups[entity]['disease_symptom']))
            complications = sorted(set(entity_relation_groups[entity]['disease_complication']))

            question = f"What are the symptoms and complications of {entity}?"
            answer = f"Symptoms: {', '.join(symptoms)}. Complications: {', '.join(complications)}."

            complex_qa_pairs.append({
                'question': question,
                'answer': answer,
                'entity': entity,
                'relation': 'disease_symptom_and_complication'
            })
            complex_questions_generated += 1

    # Example: "What bacteria cause Disease X and what antibiotics are effective against them?"
    for entity in entity_relation_groups.keys():
        if 'disease_bacteria' in entity_relation_groups[entity]:
            bacteria = entity_relation_groups[entity]['disease_bacteria']

            # For each bacteria, find effective antibiotics
            bacteria_antibiotics = []
            for bacterium in bacteria:
                if bacterium in reverse_entity_relation_groups and 'reverse_antibiotic_bacteria' in \
                        reverse_entity_relation_groups[bacterium]:
                    antibiotics = sorted(set(reverse_entity_relation_groups[bacterium]['reverse_antibiotic_bacteria']))
                    bacteria_antibiotics.append((bacterium, antibiotics))

            if bacteria_antibiotics:
                question = f"What bacteria cause {entity} and what antibiotics are effective against them?"
                answer_parts = []
                for bacterium, antibiotics in bacteria_antibiotics:
                    if antibiotics:
                        answer_parts.append(f"{bacterium} (treatable with: {', '.join(antibiotics)})")
                    else:
                        answer_parts.append(bacterium)

                answer = "The bacteria associated with this disease are: " + "; ".join(answer_parts)

                complex_qa_pairs.append({
                    'question': question,
                    'answer': answer,
                    'entity': entity,
                    'relation': 'disease_bacteria_and_antibiotics'
                })
                complex_questions_generated += 1

    # Example: "What symptoms and treatments are associated with Disease X?"
    for entity in entity_relation_groups.keys():
        if 'disease_symptom' in entity_relation_groups[entity] and 'disease_treatment' in entity_relation_groups[
            entity]:
            symptoms = sorted(set(entity_relation_groups[entity]['disease_symptom']))
            treatments = sorted(set(entity_relation_groups[entity]['disease_treatment']))

            question = f"What symptoms and treatments are associated with {entity}?"
            answer = f"Symptoms: {', '.join(symptoms)}. Treatments: {', '.join(treatments)}."

            complex_qa_pairs.append({
                'question': question,
                'answer': answer,
                'entity': entity,
                'relation': 'disease_symptom_and_treatment'
            })
            complex_questions_generated += 1

    # Example: "What are the treatment options and contraindications for Disease X?"
    for entity in entity_relation_groups.keys():
        has_treatments = 'disease_treatment' in entity_relation_groups[entity]
        has_contraindications = 'contraindication_drug' in entity_relation_groups[
            entity] or 'contraindication_situation' in entity_relation_groups[entity]

        if has_treatments and has_contraindications:
            treatments = sorted(set(entity_relation_groups[entity]['disease_treatment']))

            contraindications = []
            if 'contraindication_drug' in entity_relation_groups[entity]:
                contraindications.extend(entity_relation_groups[entity]['contraindication_drug'])
            if 'contraindication_situation' in entity_relation_groups[entity]:
                contraindications.extend(entity_relation_groups[entity]['contraindication_situation'])

            question = f"What are the treatment options and contraindications for {entity}?"
            answer = f"Treatments: {', '.join(treatments)}. Contraindications: {', '.join(sorted(set(contraindications)))}."

            complex_qa_pairs.append({
                'question': question,
                'answer': answer,
                'entity': entity,
                'relation': 'disease_treatment_and_contraindications'
            })
            complex_questions_generated += 1

    # Add complex QA pairs to the dataframe
    if complex_qa_pairs:
        complex_qa_df = pd.DataFrame(complex_qa_pairs)
        qa_df = pd.concat([qa_df, complex_qa_df], ignore_index=True)
        logger.info(f"Added {len(complex_qa_df)} complex question-answer pairs")

    logger.info(f"Total QA pairs generated: {len(qa_df)}")
    return qa_df


def save_qa_pairs(qa_df, output_xlsx):
    """
    Save the QA pairs to Excel with multiple sheets.
    """
    logger.info(f"Saving {len(qa_df)} QA pairs to {output_xlsx}...")

    try:
        with pd.ExcelWriter(output_xlsx, engine='openpyxl') as writer:
            # All QA pairs
            qa_df.to_excel(writer, sheet_name='All_QA_Pairs', index=False)

            # Split by relation type
            relation_types = qa_df['relation'].unique()
            for relation in relation_types:
                relation_df = qa_df[qa_df['relation'] == relation]
                # Excel sheet name length limit and invalid character handling
                sheet_name = re.sub(r'[\\/*\[\]:?]', '_', relation)[:30]
                relation_df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"Successfully saved QA pairs to: {os.path.abspath(output_xlsx)}")

        # Also save a backup CSV in case Excel fails
        csv_backup = output_xlsx.replace('.xlsx', '_backup.csv')
        qa_df.to_csv(csv_backup, index=False)
        logger.info(f"Created backup CSV at: {os.path.abspath(csv_backup)}")

    except Exception as e:
        logger.error(f"Error saving to Excel: {e}")
        # Fallback to CSV format
        csv_output = output_xlsx.replace('.xlsx', '.csv')
        qa_df.to_csv(csv_output, index=False)
        logger.warning(f"Saved QA pairs to CSV instead: {os.path.abspath(csv_output)}")


# Example usage
if __name__ == "__main__":
    try:
        # Hardcoded file paths
        input_file = r"C:\Users\38674\Desktop\data\抗菌药物数据\最新\呼吸道感染三元组.xlsx"
        output_file = r"C:\Users\38674\Desktop\data\抗菌药物数据\最新\respiratory_qa_pairs.xlsx"
        error_log = r"C:\Users\38674\Desktop\data\抗菌药物数据\最新\qa_conversion_errors.log"

        print(f"Input file: {input_file}")
        print(f"Output will be saved to: {output_file}")
        print(f"Error log will be saved to: {error_log}")

        qa_pairs = convert_triples_to_qa_pairs(input_file, output_file, error_log)

        if qa_pairs is not None:
            # Print some statistics
            print(f"\nSummary:")
            print(f"  Total QA pairs generated: {len(qa_pairs)}")

            # Count by relation type
            relation_counts = qa_pairs['relation'].value_counts()
            print("\nQA pairs by relation type:")
            for rel, count in relation_counts.items():
                print(f"  {rel}: {count}")

            print(f"\nConversion completed successfully!")
        else:
            print(f"\nConversion failed. Check the error log for details: {error_log}")

    except Exception as e:
        print(f"Error during script execution: {e}")
        print(traceback.format_exc())

