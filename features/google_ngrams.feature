Scenario: Build a vector space from Google Books ngrams
    Given I have Google Books co-occurrence counts
    When I build a co-occurrence matrix
    Then I should have the Google co-occurrence space file

Scenario: Apply tf-id weighting to the co-occurrence matrix
    Given I have Google Books co-occurrence counts
    When I build a co-occurrence matrix
    And I apply tf-idf weighting
    Then I should have the tf-idf weighted space file
