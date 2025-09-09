import re

class KyxParser:
    def __init__(self):
        # Regular expressions can be class attributes for better organization
        self.definitions_regex = r"Definitions\s*(?:\/\*.*?\*\/\s*)?(.*?)(?=\s*End\.)"
        self.variables_regex = r"ProgramVariables\s*(?:\/\*.*?\*\/\s*)?(.*?)(?=\s*End\.)"
        self.problem_regex = r"Problem\s*(?:\/\*.*?\*\/\s*)?(.*?)(?=\s*End\.)"
        self.predicate_regex = r"(Bool|HP|Real)\s+(\w+)\(([\w\s,]+)\)\s*(<->|::=)\s*({?[^;}]*(;[^;}]*)*}?|;)"
        self.problem_pattern = r"""
            ^\s*(?P<precondition>.*?)\s*->\s*\[
            \s*(?P<model>\{.*?\})\s*
            \*@\s*invariant\s*\((?P<invariant>.*?)\)\s*
            \]\s*(?P<safety_condition>.*?)\s*$
        """


    def parse(self, kyx_content):
        """Parses the KYX content and returns a structured dictionary."""
        kyx_content = KyxParser.remove_comments(kyx_content)
        sections = self._extract_sections(kyx_content)
        #print("sections: ", sections)
        if sections is None:
            return None  # Parsing failed

        return self._parse_sections(sections)

    def _extract_sections(self, kyx_content):
        """Extracts Definitions, ProgramVariables, and Problem sections."""

        definitions_match = re.search(self.definitions_regex, kyx_content, re.DOTALL)
        variables_match = re.search(self.variables_regex, kyx_content, re.DOTALL)
        problem_match = re.search(self.problem_regex, kyx_content, re.DOTALL)

        if not all([definitions_match, variables_match, problem_match]):
            print("Error: Could not parse all sections from the file.")
            return None

        return {
            "Definitions": definitions_match.group(1).strip(),
            "ProgramVariables": variables_match.group(1).strip(),
            "Problem": problem_match.group(1).strip(),
        }

    def _parse_sections(self, sections):
        """Parses the extracted sections into a structured dictionary."""

        parsed_spec = {}
        for section, content in sections.items():
            if section in ['Definitions', 'ProgramVariables']:
                parsed_spec[section] = self._parse_declarations(content)
            elif section == 'Problem':
                parsed_spec[section] = self._parse_problem(content)

        return parsed_spec

    def _parse_declarations(self, declarations_string):
      """Parses declaration strings (Definitions, ProgramVariables) into dictionaries."""
      declarations = {}
      for line in declarations_string.splitlines():
          line = line.strip()
          #print("line: ", line)
          if line:
              match = re.match(self.predicate_regex, line)
              if match:
                  index = re.search(r"(::=|=|<->)", line).start()
                  declarations[line[:index]] = line[index:]
              else:
                parts = line.split(';')
                for part in parts:
                    part = part.strip()
                    if part:
                        try:
                            var_type, var_name = part.split()
                            declarations[var_name] = var_type
                        except ValueError:
                            print(f"Warning: Invalid declaration format: {part}")

      print("declarations: ", declarations)

      return declarations


    def _parse_problem(self, problem_string):
      """Parses the problem statement into a structured dictionary."""

      match = re.match(self.problem_pattern, problem_string, re.DOTALL | re.VERBOSE)

      if not match:
          print("Error: Could not parse the problem statement. Check the format.")
          return {"error": "Could not parse problem statement"} # Return error indication


      parsed_data = match.groupdict()

      # Clean up and restructure the dictionary (same as before)
      cleaned_data = {
          "precondition": parsed_data["precondition"].strip(),
          "invariant": parsed_data["invariant"].strip(),
          "safety_condition": parsed_data["safety_condition"].strip(),
          "model": {
              "statements": [stmt.strip() for stmt in parsed_data["model"].strip().split(';') if stmt.strip()]
          }
      }

      # Clean up extracted strings (remove extra whitespace)
      for key, value in parsed_data.items():
          parsed_data[key] = value.strip()

      #Further processing for the model (this part is highly dependent on the specific logic)
      model_string = parsed_data["model"]
      # Example:  A very basic attempt to separate assignments and differential equations.
      # This would need significant improvement for robustness.
      model_parts = re.split(r"\s*;\s*", model_string)
      parsed_data["model"] = model_parts

      return parsed_data

    @staticmethod
    def remove_comments(text):
        """Removes C-style comments from a string."""
        # Regular expression to match C-style comments (/* ... */)
        pattern = r"/\*.*?\*/"
        return re.sub(pattern, "", text, flags=re.DOTALL)

    def to_Python(self, parsed_data):
      pass


