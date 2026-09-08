export interface paths {
    "/api/benchmarks": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Benchmarks
         * @description List the suites under the configured benchmark root.
         *
         *     A page rather than a bare array, using the same
         *     :class:`~llm_eval_lab.models.Page` shape the runs and cases listings use. A
         *     root holding more suites than one listing returns is truncated
         *     deterministically to the lexicographically first of them, and ``total``
         *     greater than the number of items is how a client learns that happened - a
         *     silently short array would read as an inventory. The same fact is also
         *     sent as :data:`LISTING_TRUNCATED_HEADER`, so a caller that wants a plain
         *     yes/no need not compare ``total`` against ``len(items)`` itself.
         *
         *     Returns an empty page when no root is configured, rather than an error: "no
         *     suites are addressable by name here" is a true and useful answer, and the
         *     attempt to address one by name reports the missing configuration precisely.
         */
        get: operations["benchmarks_api_benchmarks_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/benchmarks/validate": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Validate
         * @description Report every problem in one suite, or that it has none.
         *
         *     A ``200`` carrying ``valid: false`` rather than a ``422``, because the caller
         *     asked a question and "no" is a successful answer to it. Trying to RUN an
         *     invalid suite is what produces the ``422`` problem detail.
         */
        post: operations["validate_api_benchmarks_validate_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/benchmarks/{name}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Benchmark
         * @description Return one suite in resolved, executable form, with the digest a run records.
         *
         *     Secret-shaped values are scrubbed on the way out; see
         *     :class:`~llm_eval_lab.api.schemas.BenchmarkDetail`.
         */
        get: operations["benchmark_api_benchmarks__name__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/compare": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Compare By Query
         * @description Compare two runs under the shipped default policy.
         *
         *     The linkable form, for a dashboard that wants a comparison in a URL. A custom
         *     policy needs a body, so it needs the ``POST``.
         */
        get: operations["compare_by_query_api_compare_get"];
        put?: never;
        /**
         * Compare
         * @description Compare a candidate run against a baseline under an inline or default policy.
         */
        post: operations["compare_api_compare_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/evaluators": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Evaluators
         * @description List every registered evaluator type with the JSON Schema of its parameters.
         */
        get: operations["evaluators_api_evaluators_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/health": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Health
         * @description Report whether this process, and the database behind it, are usable.
         *
         *     The database is checked by ISSUING A QUERY. A health check that only
         *     confirms a connection object exists reports that the process is running,
         *     which is the one thing the caller could already see.
         *
         *     Answers ``503`` when the query fails, so a supervisor keys off the status
         *     code rather than parsing the body.
         */
        get: operations["health_api_health_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/models/summary": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Model Summary
         * @description Pool every stored run rollup into one row per provider and model.
         */
        get: operations["model_summary_api_models_summary_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/pricing": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Pricing
         * @description Return the price table this process costs runs against.
         *
         *     The ``content_hash`` travels with it deliberately: it is what lets somebody
         *     confirm a historical run's costs came from this exact table rather than from
         *     a later edit that kept the same version string.
         */
        get: operations["pricing_api_pricing_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/providers": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Providers
         * @description List every registered provider, its required extra and its credential VARIABLE.
         *
         *     ``credential_env`` is a variable name and ``credential_resolved`` is a
         *     boolean. No field on this response can hold a credential value, which is what
         *     makes the endpoint safe to render in a browser.
         */
        get: operations["providers_api_providers_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Runs
         * @description List runs newest first, with the total behind the page.
         */
        get: operations["list_runs_api_runs_get"];
        put?: never;
        /**
         * Create Run
         * @description Validate and start one run, returning immediately with its id.
         *
         *     The run executes as a background task owned by this process. Poll
         *     ``links.status`` until it reports a terminal status.
         */
        post: operations["create_run_api_runs_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Run
         * @description Return one run with the pass rate computed from its persisted cases.
         *
         *     ``run.config.suite_source`` is ``None``: the server's path to the suite file
         *     is not part of the answer to "what did this run do". The suite name, version
         *     and content hash beside it are.
         */
        get: operations["get_run_api_runs__id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{id}/cancel": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        get?: never;
        put?: never;
        /**
         * Cancel Run
         * @description Ask one run to stop after the cases already in flight finish.
         *
         *     ``202``, not ``200``: the run is asked to stop, and stops when its in-flight
         *     cases have been written. A run that has already finished, or that this
         *     process is not executing, is a ``409`` - answering ``202`` for an instruction
         *     nothing will carry out is worse than an error.
         */
        post: operations["cancel_run_api_runs__id__cancel_post"];
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{id}/cases": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * List Cases
         * @description List one run's cases as rows, without their model outputs.
         */
        get: operations["list_cases_api_runs__id__cases_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{id}/cases/{case_id}": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Get Case
         * @description Return one case in full: its response, its evaluations and their provenance.
         */
        get: operations["get_case_api_runs__id__cases__case_id__get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{id}/export": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Export Run
         * @description Send one run's case results as a JSON document or a CSV file.
         *
         *     The CSV column order, the JSON row shape and the JSON envelope all come from
         *     shared code, so an export taken here is comparable with one taken through
         *     ``llm-eval export``.
         *
         *     Chunked transfer, not incremental streaming: the case results are read in
         *     full and the document is rendered before the first byte goes out, so peak
         *     memory is the size of the whole export. Rendering lazily would need the
         *     repository's own async iterator to survive past its unit of work, which is a
         *     persistence-layer change rather than a routing one. The chunking still keeps
         *     the transport from holding a second copy.
         */
        get: operations["export_run_api_runs__id__export_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{id}/metrics": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Run Metrics
         * @description Return one run's aggregate rollup.
         *
         *     See :data:`MATERIALIZED_HEADER` for how a stored rollup is distinguished from
         *     one computed for this request.
         */
        get: operations["run_metrics_api_runs__id__metrics_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/runs/{id}/status": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Run Status
         * @description Report one run's progress. The endpoint a client polls after a ``202``.
         */
        get: operations["run_status_api_runs__id__status_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
    "/api/version": {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        /**
         * Version
         * @description Report the library version, the interpreter, and this API's schema version.
         */
        get: operations["version_api_version_get"];
        put?: never;
        post?: never;
        delete?: never;
        options?: never;
        head?: never;
        patch?: never;
        trace?: never;
    };
}
export type webhooks = Record<string, never>;
export interface components {
    schemas: {
        /**
         * AggregateMetrics
         * @description The computed rollup for one run.
         *
         *     ``error_rate`` is always over total attempted, whatever ``error_policy``
         *     excludes from the pass-rate denominator, and that denominator is recorded
         *     explicitly in ``n_pass_denominator`` rather than left to be re-derived. The
         *     two coverage ratios say what fraction of cases actually carried token usage
         *     and a price, so a cost figure is never read as complete when it is not.
         */
        AggregateMetrics: {
            /** By Category */
            by_category: {
                [key: string]: components["schemas"]["CategoryMetrics"];
            };
            /** By Evaluator */
            by_evaluator: {
                [key: string]: components["schemas"]["EvaluatorMetrics"];
            };
            /** By Tag */
            by_tag: {
                [key: string]: components["schemas"]["CategoryMetrics"];
            };
            computed_at: components["schemas"]["UtcDatetime"];
            cost: components["schemas"]["CostBreakdown"];
            /** Cost Coverage */
            cost_coverage: number | null;
            /** Cost Per Case */
            cost_per_case: string | null;
            /** Cost Per Successful Evaluation */
            cost_per_successful_evaluation: string | null;
            /** Error Breakdown */
            error_breakdown: {
                [key: string]: number;
            };
            /** Error Policy */
            error_policy: string;
            /** Error Rate */
            error_rate: number;
            latency: components["schemas"]["LatencyStats"];
            /** Mean Score */
            mean_score: number | null;
            /** Median Score */
            median_score: number | null;
            /** N Cases */
            n_cases: number;
            /** N Completed */
            n_completed: number;
            /** N Errors */
            n_errors: number;
            /**
             * N Pass Denominator
             * @default 0
             */
            n_pass_denominator: number;
            /**
             * N Passed
             * @default 0
             */
            n_passed: number;
            /**
             * N Score Only
             * @default 0
             */
            n_score_only: number;
            /** N Scored */
            n_scored: number;
            /** N Skipped */
            n_skipped: number;
            /** N Timeouts */
            n_timeouts: number;
            /** Pass Rate */
            pass_rate: number | null;
            /** Pass Rate Ci */
            pass_rate_ci: [
                number,
                number
            ] | null;
            /** Run Id */
            run_id: string;
            /** Token Usage Coverage */
            token_usage_coverage: number | null;
            tokens: components["schemas"]["TokenTotals"];
        };
        /**
         * ApiFieldError
         * @description One problem found in a benchmark file, as the API reports it.
         *
         *     Deliberately NOT :class:`~llm_eval_lab.models.FieldError`, and deliberately
         *     two fields shorter than it.
         *
         *     ``file`` is dropped because it is the absolute path of the file on the
         *     server, and an error response is not the place to describe the server's file
         *     system to a client.
         *
         *     ``input_repr`` is dropped because it is a repr of whatever the loader was
         *     looking at when validation failed, and that is benchmark-file content. When
         *     a required field is missing, pydantic reports the failure at the location of
         *     the FIELD and echoes the whole PARENT object as the input - so a case that
         *     omits ``id`` and carries a credential in its ``metadata`` echoed the
         *     credential back, in a body the API returned with a 200. Redacting by
         *     inspecting the location name could not see that, because the location named
         *     the missing field and not the secret beside it. Removing the member removes
         *     the class of defect, and matches what
         *     :func:`request_validation_errors` already does for request bodies. The
         *     location, the case id and the message say what is wrong; the file on the
         *     operator's disk says what it is wrong in.
         */
        ApiFieldError: {
            /** Case Id */
            case_id: string | null;
            /** Location */
            location: string;
            /** Message */
            message: string;
        };
        /**
         * BenchmarkDetail
         * @description What ``GET /api/benchmarks/{name}`` returns: the resolved suite and its digest.
         *
         *     The resolved form, because that is what a run actually executes and what
         *     ``suite_hash`` addresses. The absolute path of the file is deliberately
         *     absent: the client addressed it by name and does not need the server's
         *     directory layout to address it again.
         *
         *     Secret-shaped values in the two free-form dictionaries a benchmark author
         *     controls are replaced before the suite is returned, by the same scrubber the
         *     storage layer uses. ``redacted`` says whether anything was replaced, so
         *     ``suite_hash`` - which addresses the ORIGINAL file - is never mistaken for a
         *     digest of the body beside it.
         */
        BenchmarkDetail: {
            /** Name */
            name: string;
            /** Redacted */
            redacted: boolean;
            suite: components["schemas"]["ResolvedSuite"];
            /** Suite Hash */
            suite_hash: string;
        };
        /**
         * BenchmarkEntry
         * @description One row of ``GET /api/benchmarks``.
         *
         *     A suite that does not load is listed with ``valid: false`` and the number of
         *     problems in it, rather than omitted. A hidden broken file looks exactly like
         *     a missing one, and sends the operator looking for the wrong fault.
         */
        BenchmarkEntry: {
            /** Evaluator Types */
            evaluator_types: string[];
            /** N Cases */
            n_cases: number | null;
            /** N Errors */
            n_errors: number;
            /** Name */
            name: string;
            /** Suite Hash */
            suite_hash: string | null;
            /** Suite Name */
            suite_name: string | null;
            /** Valid */
            valid: boolean;
            /** Version */
            version: string | null;
        };
        /**
         * CancelResponse
         * @description The ``202`` answer to ``POST /api/runs/{id}/cancel``.
         *
         *     ``202`` rather than ``200`` because cancellation is cooperative: the run is
         *     asked to stop and finishes the cases already in flight. ``outcome`` says what
         *     the request actually achieved, and ``status`` is the run's status at the
         *     moment the request was handled, not the status it will end at.
         */
        CancelResponse: {
            /** Detail */
            detail: string;
            /** Outcome */
            outcome: string;
            /** Run Id */
            run_id: string;
            status: components["schemas"]["RunStatus"];
        };
        /**
         * CaseDelta
         * @description How one case moved between the baseline run and the candidate run.
         */
        CaseDelta: {
            /** Baseline Passed */
            baseline_passed: boolean | null;
            /** Baseline Score */
            baseline_score: number | null;
            /** Candidate Passed */
            candidate_passed: boolean | null;
            /** Candidate Score */
            candidate_score: number | null;
            /** Case Id */
            case_id: string;
            /** Category */
            category?: string | null;
            /** Delta */
            delta: number | null;
        };
        /**
         * CaseResult
         * @description Everything recorded about one case in one run.
         */
        CaseResult: {
            /**
             * Attempts
             * @default 1
             */
            attempts: number;
            /** Case Hash */
            case_hash: string;
            /** Case Id */
            case_id: string;
            /** Category */
            category?: string | null;
            completed_at: components["schemas"]["UtcDatetime"] | null;
            cost: components["schemas"]["CostBreakdown"] | null;
            error?: components["schemas"]["ProviderErrorInfo"] | null;
            /**
             * Evaluations
             * @default []
             */
            evaluations: components["schemas"]["EvaluationResult"][];
            /** Passed */
            passed: boolean | null;
            response: components["schemas"]["ModelResponse"] | null;
            /** Run Id */
            run_id: string;
            /** Score */
            score: number | null;
            started_at: components["schemas"]["UtcDatetime"] | null;
            status: components["schemas"]["CaseStatus"];
            /**
             * Tags
             * @default []
             */
            tags: string[];
        };
        /**
         * CaseResultSummary
         * @description One row of ``GET /api/runs/{id}/cases``.
         *
         *     The model's OUTPUT is deliberately absent. A page of fifty cases is a table,
         *     and whole responses belong behind the per-case detail endpoint, which returns
         *     the full :class:`~llm_eval_lab.models.CaseResult` with its evaluations and
         *     their judge provenance.
         */
        CaseResultSummary: {
            /** Attempts */
            attempts: number;
            /** Case Hash */
            case_hash: string;
            /** Case Id */
            case_id: string;
            /** Category */
            category: string | null;
            /** Completed At */
            completed_at: string | null;
            /** Error Kind */
            error_kind: string | null;
            /** Latency Ms */
            latency_ms: number | null;
            /** Passed */
            passed: boolean | null;
            /** Priced */
            priced: boolean | null;
            /** Run Id */
            run_id: string;
            /** Score */
            score: number | null;
            /** Started At */
            started_at: string | null;
            status: components["schemas"]["CaseStatus"];
            /** Tags */
            tags: string[];
            /** Total Cost */
            total_cost: string | null;
            /** Total Tokens */
            total_tokens: number | null;
        };
        /**
         * CaseSelection
         * @description Exactly which cases the run executed, and how they were chosen.
         */
        CaseSelection: {
            /**
             * Case Ids
             * @default []
             */
            case_ids: string[];
            /** Limit */
            limit?: number | null;
            /** N Selected */
            n_selected: number;
            /** Sample */
            sample?: number | null;
            /** Sample Seed */
            sample_seed?: number | null;
            /**
             * Tags
             * @default []
             */
            tags: string[];
        };
        /**
         * CaseStatus
         * @description Lifecycle state of a single case within a run.
         * @enum {string}
         */
        CaseStatus: "ok" | "error" | "timeout" | "skipped" | "cancelled";
        /**
         * CategoryMetrics
         * @description Pass rate and mean score for one category or tag bucket.
         */
        CategoryMetrics: {
            /** Mean Score */
            mean_score: number | null;
            /** N */
            n: number;
            /** N Passed */
            n_passed: number;
            /** Pass Rate */
            pass_rate: number | null;
            /** Pass Rate Ci */
            pass_rate_ci: [
                number,
                number
            ] | null;
        };
        /**
         * ChatMessage
         * @description One turn of a conversation, in provider-neutral form.
         */
        ChatMessage: {
            /** Content */
            content: string;
            /** Name */
            name?: string | null;
            /**
             * Role
             * @enum {string}
             */
            role: "system" | "user" | "assistant";
        };
        /**
         * CheckOutcome
         * @description The result of one threshold check, with its advisory statistics.
         *
         *     ``p_value`` is the McNemar exact test and is populated only for paired
         *     pass-rate comparisons. ``confidence_interval`` is the Newcombe Method-10
         *     interval on the paired pass-rate delta, and is ``None`` for every other
         *     metric and in unpaired mode, where the two Wilson intervals in
         *     ``baseline_ci`` and ``candidate_ci`` take its place instead.
         */
        CheckOutcome: {
            /** Baseline */
            baseline: number | null;
            /** Baseline Ci */
            baseline_ci?: [
                number,
                number
            ] | null;
            /** Candidate */
            candidate: number | null;
            /** Candidate Ci */
            candidate_ci?: [
                number,
                number
            ] | null;
            /** Confidence Interval */
            confidence_interval: [
                number,
                number
            ] | null;
            /** Delta */
            delta: number | null;
            direction: components["schemas"]["ThresholdDirection"];
            /** Intervals Overlap */
            intervals_overlap?: boolean | null;
            /** Label */
            label: string;
            /** Metric */
            metric: string;
            /** N Baseline */
            n_baseline: number;
            /** N Candidate */
            n_candidate: number;
            /** N Paired */
            n_paired: number | null;
            /** Note */
            note: string | null;
            /** P Value */
            p_value: number | null;
            /** Relative Delta */
            relative_delta: number | null;
            /** Significant */
            significant: boolean | null;
            status: components["schemas"]["CheckStatus"];
            /** Threshold Value */
            threshold_value: number | null;
            /** Violated Bound */
            violated_bound: string | null;
        };
        /**
         * CheckStatus
         * @description Outcome of a single regression threshold check.
         * @enum {string}
         */
        CheckStatus: "passed" | "failed" | "warning" | "insufficient_data" | "missing_metric";
        /**
         * CompareRequest
         * @description Body of ``POST /api/compare``.
         *
         *     ``thresholds`` is an inline policy, not a path: the API accepts no
         *     server-side file paths. Omitting it uses the policy shipped with the package,
         *     which is the same default ``llm-eval compare`` uses.
         */
        CompareRequest: {
            baseline_run_id: components["schemas"]["ShortText"];
            candidate_run_id: components["schemas"]["ShortText"];
            thresholds?: components["schemas"]["RegressionThresholds"] | null;
        };
        /**
         * CostBreakdown
         * @description What one unit of work cost, and the exact prices that produced it.
         *
         *     ``judge_cost`` is tracked separately and is never folded into
         *     ``total_cost``: model-graded evaluation is a distinct line item, and
         *     hiding it inside the generation cost makes a judge look free.
         */
        CostBreakdown: {
            /**
             * Currency
             * @default USD
             * @constant
             */
            currency: "USD";
            /** Input Cost */
            input_cost: string | null;
            /** Input Per Mtok */
            input_per_mtok?: string | null;
            /** Judge Cost */
            judge_cost?: string | null;
            /** Output Cost */
            output_cost: string | null;
            /** Output Per Mtok */
            output_per_mtok?: string | null;
            /** Price Table Id */
            price_table_id: string;
            /** Price Table Version */
            price_table_version: string;
            /** Priced */
            priced: boolean;
            /** Total Cost */
            total_cost: string | null;
            /** Unpriced Reason */
            unpriced_reason?: string | null;
        };
        /**
         * CreateRunRequest
         * @description Body of ``POST /api/runs``.
         *
         *     ``provider`` is the frozen :class:`~llm_eval_lab.models.ProviderConfig`,
         *     used here rather than re-declared, because its validators are the security
         *     control: it carries ``api_key_env`` (a NAME) and its recursive validator
         *     refuses any option key that looks like a place somebody put a credential.
         *     Re-declaring the shape here would be re-implementing that check, badly.
         */
        CreateRunRequest: {
            /**
             * Case Ids
             * @default []
             */
            case_ids: components["schemas"]["ShortText"][];
            /** Case Timeout S */
            case_timeout_s?: number | null;
            /** Concurrency */
            concurrency?: number | null;
            /**
             * Error Policy
             * @default exclude
             * @enum {string}
             */
            error_policy: "exclude" | "fail";
            label?: components["schemas"]["ShortText"] | null;
            /** Limit */
            limit?: number | null;
            /**
             * Max Cost
             * @description Running spend cap for THIS run, in the price table's currency; the runner cancels the run once accumulated known cost exceeds it. This per-run ceiling is the only server-side spend limit: the deployment imposes no cap of its own, so omitting this launches a run with no budget. The process will not execute more than a fixed number of runs at once, which bounds concurrent spend but not the spend of any single run.
             */
            max_cost?: number | string | null;
            /** Notes */
            notes?: string | null;
            provider: components["schemas"]["ProviderConfig-Input"];
            /** Sample */
            sample?: number | null;
            /** Sample Seed */
            sample_seed?: number | null;
            /** Seed */
            seed?: number | null;
            /**
             * Select Tags
             * @default []
             */
            select_tags: components["schemas"]["ShortText"][];
            suite: components["schemas"]["SuiteName"];
            /**
             * Tags
             * @default []
             */
            tags: components["schemas"]["ShortText"][];
        };
        /**
         * CreateRunResponse
         * @description The ``202`` answer to ``POST /api/runs``: an id and where to poll it.
         */
        CreateRunResponse: {
            links: components["schemas"]["RunLinks"];
            /** Run Id */
            run_id: string;
            status: components["schemas"]["RunStatus"];
        };
        /**
         * EvaluationErrorInfo
         * @description Why an evaluator could not produce a verdict.
         */
        EvaluationErrorInfo: {
            /** Exception Type */
            exception_type?: string | null;
            /**
             * Kind
             * @enum {string}
             */
            kind: "config" | "runtime" | "provider" | "parse" | "timeout";
            /** Message */
            message: string;
            provider_error?: components["schemas"]["ProviderErrorInfo"] | null;
        };
        /**
         * EvaluationResult
         * @description The outcome of applying one evaluator to one model response.
         *
         *     The rule, because three lanes need the same one: ``status`` reports whether
         *     the **evaluator** reached a conclusion, ``passed`` reports whether the
         *     **case** met the bar. A score-only evaluator - a judge with no
         *     ``pass_threshold`` - returns ``status=PASSED`` with ``passed=None`` and a
         *     populated ``score``. That pairing, and only that pairing, is what
         *     ``n_score_only`` counts. Pass/fail counting keys off ``passed``, never off
         *     ``status``, so a score-only judgment is never silently read as a failure.
         */
        EvaluationResult: {
            /**
             * Duration Ms
             * @default 0
             */
            duration_ms: number;
            error?: components["schemas"]["EvaluationErrorInfo"] | null;
            /** Evaluator Id */
            evaluator_id: string;
            /** Evaluator Type */
            evaluator_type: string;
            /** Explanation */
            explanation?: string | null;
            judge?: components["schemas"]["JudgeProvenance"] | null;
            /** Metadata */
            metadata?: {
                [key: string]: components["schemas"]["JSONValue-Output"];
            };
            /** Passed */
            passed: boolean | null;
            /** Raw Score */
            raw_score?: number | null;
            /**
             * Required
             * @default true
             */
            required: boolean;
            /** Score */
            score?: number | null;
            status: components["schemas"]["EvaluationStatus"];
            /**
             * Weight
             * @default 1
             */
            weight: number;
        };
        /**
         * EvaluationStatus
         * @description Outcome of one evaluator applied to one model response.
         * @enum {string}
         */
        EvaluationStatus: "passed" | "failed" | "skipped" | "error";
        /**
         * EvaluatorEntry
         * @description One row of ``GET /api/evaluators``.
         */
        EvaluatorEntry: {
            /** Is Model Graded */
            is_model_graded: boolean;
            /** Needs Expected */
            needs_expected: boolean;
            /** Params Schema */
            params_schema: {
                [key: string]: unknown;
            };
            /** Summary */
            summary: string;
            /** Type */
            type: string;
        };
        /**
         * EvaluatorMetrics
         * @description Per-evaluator rollup for one run.
         */
        EvaluatorMetrics: {
            /** Evaluator Id */
            evaluator_id: string;
            /** Evaluator Type */
            evaluator_type: string;
            /** Mean Score */
            mean_score: number | null;
            /** Median Score */
            median_score: number | null;
            /** N */
            n: number;
            /** N Errors */
            n_errors: number;
            /**
             * N Pass Denominator
             * @default 0
             */
            n_pass_denominator: number;
            /** N Passed */
            n_passed: number;
            /**
             * N Score Only
             * @default 0
             */
            n_score_only: number;
            /** Pass Rate */
            pass_rate: number | null;
        };
        /**
         * EvaluatorSpec
         * @description One evaluator to apply to a case, and how its outcome is treated.
         */
        EvaluatorSpec: {
            /** Id */
            id?: string | null;
            /**
             * On Error
             * @default error
             * @enum {string}
             */
            on_error: "error" | "fail" | "skip";
            /** Params */
            params?: {
                [key: string]: components["schemas"]["JSONValue-Output"];
            };
            /**
             * Required
             * @default true
             */
            required: boolean;
            /** Type */
            type: string;
            /**
             * Weight
             * @default 1
             */
            weight: number;
        };
        /**
         * FinishReason
         * @description Why a provider stopped producing output.
         * @enum {string}
         */
        FinishReason: "stop" | "length" | "content_filter" | "tool_use" | "error" | "unknown";
        /**
         * GenerationParams
         * @description Decoding parameters, normalized across providers.
         *
         *     A provider translates these into its own vocabulary and drops what it does
         *     not support; `extra` carries vendor-specific knobs that have no neutral
         *     spelling.
         */
        GenerationParams: {
            /** Extra */
            extra?: {
                [key: string]: components["schemas"]["JSONValue-Output"];
            };
            /** Max Output Tokens */
            max_output_tokens?: number | null;
            /**
             * Response Format
             * @default text
             * @enum {string}
             */
            response_format: "text" | "json_object";
            /** Seed */
            seed?: number | null;
            /**
             * Stop
             * @default []
             */
            stop: string[];
            /** Temperature */
            temperature?: number | null;
            /** Top P */
            top_p?: number | null;
        };
        /**
         * HealthResponse
         * @description What ``GET /api/health`` reports.
         *
         *     ``database`` is the result of an actual query, not of holding a connection
         *     object: a health check that never touches the database reports the process
         *     is alive, which is the one thing nobody was worried about.
         *
         *     ``runs_in_flight`` is this process's own count: runs it is executing plus
         *     runs it has reserved a launch slot for, out of
         *     :data:`~llm_eval_lab.services.run_manager.MAX_CONCURRENT_RUNS`. A
         *     supervisor watching this number sees load before it sees a ``429``.
         */
        HealthResponse: {
            /**
             * Database
             * @enum {string}
             */
            database: "ok" | "error";
            /** Runs In Flight */
            runs_in_flight: number;
            /**
             * Status
             * @enum {string}
             */
            status: "ok" | "degraded";
            /** Version */
            version: string;
        };
        "JSONValue-Input": unknown;
        "JSONValue-Output": unknown;
        /**
         * JudgeProvenance
         * @description Everything needed to audit or reproduce a judgment.
         *
         *     ``self_preference_risk`` and ``high_judge_variance`` are advisory: they are
         *     always visible and never silently suppress or gate a result.
         */
        JudgeProvenance: {
            /** Aggregation */
            aggregation: string;
            /** Agreement */
            agreement: number | null;
            /** Cost Usd */
            cost_usd: string | null;
            /** Dispersion */
            dispersion: number | null;
            /**
             * High Judge Variance
             * @default false
             */
            high_judge_variance: boolean;
            /** Judge Models */
            judge_models: string[];
            /**
             * Parse Failures
             * @default 0
             */
            parse_failures: number;
            /** Raw Outputs */
            raw_outputs: string[];
            /** Rendered Prompts */
            rendered_prompts: string[];
            /** Rubric Hash */
            rubric_hash: string;
            scale: components["schemas"]["JudgeScale"];
            /**
             * Self Preference Risk
             * @default false
             */
            self_preference_risk: boolean;
            /** Template Id */
            template_id: string;
            usage: components["schemas"]["TokenUsage"];
            /** Verdicts */
            verdicts: components["schemas"]["JudgeVerdict"][];
        };
        /**
         * JudgeScale
         * @description The raw scale a judge scores on, and how it maps onto 0.0..1.0.
         */
        JudgeScale: {
            /**
             * Kind
             * @default integer
             * @enum {string}
             */
            kind: "integer" | "continuous" | "binary";
            /** Labels */
            labels?: {
                [key: string]: string;
            };
            /**
             * Maximum
             * @default 5
             */
            maximum: number;
            /**
             * Minimum
             * @default 1
             */
            minimum: number;
        };
        /**
         * JudgeVerdict
         * @description The structured output the judge model is required to produce.
         */
        JudgeVerdict: {
            /** Confidence */
            confidence?: number | null;
            /** Per Criterion */
            per_criterion?: {
                [key: string]: number;
            };
            /** Reasoning */
            reasoning: string;
            /** Score */
            score: number;
            /**
             * Violations
             * @default []
             */
            violations: string[];
        };
        /**
         * LatencyStats
         * @description Latency distribution over SUCCESSFULLY COMPLETED cases only.
         *
         *     Errored cases with no measured latency are excluded, and ``n`` is what
         *     makes that exclusion visible next to every percentile.
         */
        LatencyStats: {
            /**
             * Low Confidence
             * @default []
             */
            low_confidence: string[];
            /** Max Ms */
            max_ms: number | null;
            /** Mean Ms */
            mean_ms: number | null;
            /** N */
            n: number;
            /** P50 Ms */
            p50_ms: number | null;
            /** P90 Ms */
            p90_ms: number | null;
            /** P95 Ms */
            p95_ms: number | null;
            /** P99 Ms */
            p99_ms: number | null;
        };
        /**
         * MetricCheck
         * @description One threshold applied to one metric. Pure data; it evaluates nothing.
         *
         *     Bound semantics, which the phase 4 engine in ``reporting/regression.py``
         *     implements and which are carried in the field descriptions so there is one
         *     written source: ``min_value`` is an INCLUSIVE floor, violated when the
         *     candidate is strictly below it. ``max_value`` is an EXCLUSIVE ceiling,
         *     violated when the candidate is greater than or equal to it. That asymmetry
         *     is what makes "error rate must remain below 1%" expressible as
         *     ``max_value: 0.01``, where a candidate of exactly ``0.01`` fails.
         *
         *     Evaluation deliberately does not live here. Gate semantics are engine
         *     behaviour, not contract shape, and a frozen model carrying them would force
         *     every later correction through the frozen-file protocol.
         */
        MetricCheck: {
            /** @description Which way this metric has to move to be an improvement. Descriptive: it labels the metric for rendering and arrow direction. The bounds below are already directional, so an engine must not derive a second directional rule from this field. */
            direction: components["schemas"]["ThresholdDirection"];
            /** Label */
            label?: string | null;
            /** Max Absolute Decrease */
            max_absolute_decrease?: number | null;
            /** Max Absolute Increase */
            max_absolute_increase?: number | null;
            /** Max Relative Decrease */
            max_relative_decrease?: number | null;
            /** Max Relative Increase */
            max_relative_increase?: number | null;
            /**
             * Max Value
             * @description EXCLUSIVE ceiling. Violated when the candidate is greater than or equal to it, so a candidate exactly equal to max_value FAILS. That asymmetry is what makes 'error rate must remain below 1%' expressible as max_value: 0.01.
             */
            max_value?: number | null;
            /** Metric */
            metric: string;
            /**
             * Min Samples
             * @default 1
             */
            min_samples: number;
            /**
             * Min Value
             * @description INCLUSIVE floor. Violated when the candidate is strictly below it, so a candidate exactly equal to min_value passes.
             */
            min_value?: number | null;
            /**
             * Require Significant
             * @default false
             */
            require_significant: boolean;
            /**
             * Severity
             * @default error
             * @enum {string}
             */
            severity: "error" | "warning";
        };
        /**
         * ModelResponse
         * @description The normalized result of one case's generation, successful or failed.
         *
         *     The error invariant is enforced here rather than by convention: a response
         *     either carries output text and no error, or an error and no output text.
         *     There is no third shape, so no consumer has to guess.
         */
        ModelResponse: {
            /**
             * Attempts
             * @default 1
             */
            attempts: number;
            completed_at: components["schemas"]["UtcDatetime"];
            error?: components["schemas"]["ProviderErrorInfo"] | null;
            finish_reason: components["schemas"]["FinishReason"];
            /** Latency Ms */
            latency_ms: number;
            /** Model */
            model: string;
            /** Output Text */
            output_text: string | null;
            /** Provider */
            provider: string;
            /** Provider Request Id */
            provider_request_id?: string | null;
            /** Raw */
            raw?: {
                [key: string]: components["schemas"]["JSONValue-Output"];
            };
            /** Requested Model */
            requested_model: string;
            started_at: components["schemas"]["UtcDatetime"];
            /** Total Latency Ms */
            total_latency_ms: number;
            /**
             * Truncated
             * @default false
             */
            truncated: boolean;
            /** @default {} */
            usage: components["schemas"]["TokenUsage"];
        };
        /**
         * ModelSummaryRow
         * @description One row of ``GET /api/models/summary``.
         *
         *     Every field says what it was computed over, because a leaderboard that shows
         *     a pass rate without a denominator invites a comparison the data does not
         *     support.
         */
        ModelSummaryRow: {
            /**
             * Cost Coverage
             * @description Case-weighted fraction of cases that carried a price, across these runs.
             */
            cost_coverage?: number | null;
            /**
             * Cost Per Case
             * @description Summed cost divided by the summed COMPLETED cases, the same definition a single run's metrics use, so cost_per_case * n_cases reconciles against total_cost. Cases the price table could not cover contribute to the denominator and not to the numerator, so this figure is a lower bound whenever cost_coverage is below 1: read the two together.
             */
            cost_per_case?: string | null;
            /** Last Run At */
            last_run_at: string | null;
            /**
             * Median Run P95 Ms
             * @description Median of the per-run p95 latencies. NOT a pooled percentile over cases: per-case latencies are not loaded by this endpoint, so this is a weaker statement than a p95 and is named for what it is. Read it with n_runs_with_latency.
             */
            median_run_p95_ms?: number | null;
            /** Model */
            model: string;
            /** N Cases */
            n_cases: number;
            /** N Pass Denominator */
            n_pass_denominator: number;
            /** N Passed */
            n_passed: number;
            /** N Runs */
            n_runs: number;
            /** N Runs With Latency */
            n_runs_with_latency: number;
            /**
             * Pass Rate
             * @description Pooled over every stored run of this model, not an average of run rates.
             */
            pass_rate?: number | null;
            /**
             * Pass Rate Ci
             * @description Wilson score interval at 95% over the pooled numerator and denominator.
             */
            pass_rate_ci?: [
                number,
                number
            ] | null;
            /** Provider */
            provider: string;
            /** Total Cost */
            total_cost: string | null;
        };
        /** Page[BenchmarkEntry] */
        Page_BenchmarkEntry_: {
            /** Items */
            items: components["schemas"]["BenchmarkEntry"][];
            /** Limit */
            limit: number;
            /** Offset */
            offset: number;
            /** Total */
            total: number;
        };
        /** Page[CaseResultSummary] */
        Page_CaseResultSummary_: {
            /** Items */
            items: components["schemas"]["CaseResultSummary"][];
            /** Limit */
            limit: number;
            /** Offset */
            offset: number;
            /** Total */
            total: number;
        };
        /** Page[RunSummary] */
        Page_RunSummary_: {
            /** Items */
            items: components["schemas"]["RunSummary"][];
            /** Limit */
            limit: number;
            /** Offset */
            offset: number;
            /** Total */
            total: number;
        };
        /**
         * PluginRecord
         * @description Provenance of one opt-in evaluator plugin that was loaded for this run.
         */
        PluginRecord: {
            /** Distribution */
            distribution: string;
            /** Entry Point */
            entry_point: string;
            /** Evaluator Types */
            evaluator_types: string[];
            /** Version */
            version: string;
        };
        /**
         * PriceEntry
         * @description Unit prices for one provider/model pairing.
         */
        PriceEntry: {
            /** Cached Input Per Mtok */
            cached_input_per_mtok?: string | null;
            /** Effective From */
            effective_from?: string | null;
            /** Input Per Mtok */
            input_per_mtok: string;
            /**
             * Match
             * @default exact
             * @enum {string}
             */
            match: "exact" | "prefix" | "regex";
            /** Model */
            model: string;
            /** Notes */
            notes?: string | null;
            /** Output Per Mtok */
            output_per_mtok: string;
            /** Provider */
            provider: string;
        };
        /**
         * PriceTable
         * @description A versioned, content-addressed set of price entries.
         */
        PriceTable: {
            /** Content Hash */
            content_hash: string;
            /**
             * Currency
             * @default USD
             * @constant
             */
            currency: "USD";
            /** Id */
            id: string;
            /** Models */
            models: components["schemas"]["PriceEntry"][];
            /** Source */
            source?: string | null;
            /** Version */
            version: string;
        };
        /**
         * ProblemDetail
         * @description An RFC 9457 problem detail, with this API's extension members.
         *
         *     Every extension is optional and omitted when absent, so a client can read
         *     ``status`` and ``title`` from any error and reach for the rest only when it
         *     knows the ``type``.
         */
        ProblemDetail: {
            /** Correlation Id */
            correlation_id?: string | null;
            /** Detail */
            detail: string;
            /** Env Var */
            env_var?: string | null;
            /** Errors */
            errors?: components["schemas"]["ApiFieldError"][] | null;
            /** Extra */
            extra?: string | null;
            /** Instance */
            instance?: string | null;
            /** Status */
            status: number;
            /** Title */
            title: string;
            /**
             * Type
             * @default about:blank
             */
            type: string;
        };
        /**
         * ProviderConfig
         * @description Fully describes how to reach a model.
         *
         *     Contains NO secret material, by construction: only the NAME of the
         *     environment variable holding the credential is recorded, so this object is
         *     safe to persist, log and return over the API verbatim.
         */
        "ProviderConfig-Input": {
            /** Api Key Env */
            api_key_env?: string | null;
            /** Base Url */
            base_url?: string | null;
            /** Connect Timeout S */
            connect_timeout_s?: number | null;
            /** Label */
            label?: string | null;
            /**
             * Max Retries
             * @default 3
             */
            max_retries: number;
            /** Model */
            model: string;
            /** Options */
            options?: {
                [key: string]: components["schemas"]["JSONValue-Input"];
            };
            /** Organization */
            organization?: string | null;
            /** Provider */
            provider: string;
            /**
             * Timeout S
             * @default 60
             */
            timeout_s: number;
        };
        /**
         * ProviderConfig
         * @description Fully describes how to reach a model.
         *
         *     Contains NO secret material, by construction: only the NAME of the
         *     environment variable holding the credential is recorded, so this object is
         *     safe to persist, log and return over the API verbatim.
         */
        "ProviderConfig-Output": {
            /** Api Key Env */
            api_key_env?: string | null;
            /** Base Url */
            base_url?: string | null;
            /** Connect Timeout S */
            connect_timeout_s?: number | null;
            /** Label */
            label?: string | null;
            /**
             * Max Retries
             * @default 3
             */
            max_retries: number;
            /** Model */
            model: string;
            /** Options */
            options?: {
                [key: string]: components["schemas"]["JSONValue-Output"];
            };
            /** Organization */
            organization?: string | null;
            /** Provider */
            provider: string;
            /**
             * Timeout S
             * @default 60
             */
            timeout_s: number;
        };
        /**
         * ProviderEntry
         * @description One row of ``GET /api/providers``.
         *
         *     ``credential_env`` is an environment variable NAME and ``credential_resolved``
         *     says only whether that variable is set. There is no field on this object that
         *     could hold a credential value, which is what makes the endpoint safe to call
         *     from a browser.
         */
        ProviderEntry: {
            /** Available */
            available: boolean;
            /** Credential Env */
            credential_env: string | null;
            /** Credential Resolved */
            credential_resolved: boolean | null;
            /** Extra */
            extra: string | null;
            /** Models */
            models: string[];
            /** Name */
            name: string;
            /** Requires Credential */
            requires_credential: boolean;
            /** Summary */
            summary: string;
        };
        /**
         * ProviderErrorInfo
         * @description Structured, persistable record of a failed provider call.
         *
         *     ``message`` has already been scrubbed by the provider layer, and
         *     ``exception_type`` carries a class name only; traceback text never reaches
         *     this object because it reaches storage, logs and the HTTP API.
         */
        ProviderErrorInfo: {
            /**
             * Attempts
             * @default 1
             */
            attempts: number;
            /** Exception Type */
            exception_type?: string | null;
            kind: components["schemas"]["ProviderErrorKind"];
            /** Message */
            message: string;
            /** Provider */
            provider: string;
            /** Retry After S */
            retry_after_s?: number | null;
            /**
             * Retryable
             * @default false
             */
            retryable: boolean;
            /** Status Code */
            status_code?: number | null;
            /** Vendor Code */
            vendor_code?: string | null;
        };
        /**
         * ProviderErrorKind
         * @description Vendor-neutral classification of a single failed provider attempt.
         * @enum {string}
         */
        ProviderErrorKind: "authentication" | "permission" | "invalid_request" | "not_found" | "context_length" | "content_filter" | "unsupported" | "not_installed" | "quota" | "cancelled" | "unknown" | "rate_limit" | "timeout" | "connection" | "server";
        /**
         * RegressionReport
         * @description A complete baseline-versus-candidate comparison.
         *
         *     ``mode`` is ``paired`` when the two runs share case ids and ``unpaired``
         *     otherwise. Thresholds evaluate point estimates in both modes, so the gate
         *     verdict never depends on the mode.
         */
        RegressionReport: {
            /** Baseline Label */
            baseline_label: string | null;
            /** Baseline Run Id */
            baseline_run_id: string;
            /** Baseline Suite Hash */
            baseline_suite_hash: string;
            /** Candidate Label */
            candidate_label: string | null;
            /** Candidate Run Id */
            candidate_run_id: string;
            /** Candidate Suite Hash */
            candidate_suite_hash: string;
            /** Changed Cases */
            changed_cases: string[];
            /** Checks */
            checks: components["schemas"]["CheckOutcome"][];
            /** Comparable */
            comparable: boolean;
            generated_at: components["schemas"]["UtcDatetime"];
            /** Incomparable Reason */
            incomparable_reason: string | null;
            /** Intervals Overlap */
            intervals_overlap?: boolean | null;
            /**
             * Mode
             * @enum {string}
             */
            mode: "paired" | "unpaired";
            /** Only In Baseline */
            only_in_baseline: string[];
            /** Only In Candidate */
            only_in_candidate: string[];
            /** Paired Case Count */
            paired_case_count: number;
            /**
             * Schema Version
             * @default 1
             * @constant
             */
            schema_version: 1;
            /** Suite Hash Match */
            suite_hash_match: boolean;
            summary: components["schemas"]["RegressionSummary"];
            /** Thresholds Id */
            thresholds_id: string;
            /** Thresholds Version */
            thresholds_version: string;
            verdict: components["schemas"]["Verdict"];
        };
        /**
         * RegressionSummary
         * @description Which cases flipped, and the largest score drops behind the headline.
         */
        RegressionSummary: {
            /** Largest Score Drops */
            largest_score_drops: components["schemas"]["CaseDelta"][];
            /** N Flipped */
            n_flipped: number;
            /** Newly Failing */
            newly_failing: string[];
            /** Newly Passing */
            newly_passing: string[];
            /** Still Failing */
            still_failing: string[];
        };
        /**
         * RegressionThresholds
         * @description A named, versioned threshold policy.
         */
        RegressionThresholds: {
            /** Checks */
            checks: components["schemas"]["MetricCheck"][];
            /**
             * Confidence Level
             * @default 0.95
             */
            confidence_level: number;
            /**
             * Fail On Missing Metric
             * @default true
             */
            fail_on_missing_metric: boolean;
            /**
             * Id
             * @default default
             */
            id: string;
            /**
             * Min Paired Cases
             * @default 1
             */
            min_paired_cases: number;
            /**
             * Paired
             * @default true
             */
            paired: boolean;
            /**
             * Require Same Suite
             * @default true
             */
            require_same_suite: boolean;
            /**
             * Version
             * @default 1
             */
            version: string;
        };
        /**
         * ResolvedCase
         * @description A case with suite defaults merged in.
         *
         *     This is what executes and what is hashed; nothing downstream re-reads the
         *     authored form.
         */
        ResolvedCase: {
            /** Case Hash */
            case_hash: string;
            /** Category */
            category: string | null;
            /** Evaluators */
            evaluators: components["schemas"]["EvaluatorSpec"][];
            expected: components["schemas"]["JSONValue-Output"];
            /** Id */
            id: string;
            /** Messages */
            messages: components["schemas"]["ChatMessage"][];
            /** Metadata */
            metadata: {
                [key: string]: components["schemas"]["JSONValue-Output"];
            };
            params: components["schemas"]["GenerationParams"];
            /** System */
            system: string | null;
            /** Tags */
            tags: string[];
            /** Weight */
            weight: number;
        };
        /**
         * ResolvedSuite
         * @description A whole suite in resolved, executable, content-addressed form.
         */
        ResolvedSuite: {
            /** Cases */
            cases: components["schemas"]["ResolvedCase"][];
            /** Metadata */
            metadata: {
                [key: string]: components["schemas"]["JSONValue-Output"];
            };
            /** Name */
            name: string;
            /** Suite Hash */
            suite_hash: string;
            /** Version */
            version: string;
        };
        /**
         * RetryPolicy
         * @description Bounded exponential backoff with jitter, owned by the runner.
         *
         *     Providers perform exactly one attempt per call (decision D4), so this
         *     policy is the single place retry behaviour is described.
         */
        RetryPolicy: {
            /**
             * Base Delay S
             * @default 0.5
             */
            base_delay_s: number;
            /**
             * Jitter
             * @default full
             * @enum {string}
             */
            jitter: "full" | "equal" | "none";
            /**
             * Max Attempts
             * @default 4
             */
            max_attempts: number;
            /**
             * Max Delay S
             * @default 30
             */
            max_delay_s: number;
            /**
             * Max Total Delay S
             * @default 120
             */
            max_total_delay_s: number;
            /**
             * Respect Retry After
             * @default true
             */
            respect_retry_after: boolean;
        };
        /**
         * Run
         * @description One execution of a suite against one model.
         *
         *     ``id`` is a canonical lowercase uuid4 string. Sortability is supplied by
         *     the ``runs(created_at desc)`` index, not by the id.
         */
        Run: {
            /** Baseline Run Id */
            baseline_run_id?: string | null;
            completed_at: components["schemas"]["UtcDatetime"] | null;
            config: components["schemas"]["RunConfig"];
            created_at: components["schemas"]["UtcDatetime"];
            /** Error */
            error?: string | null;
            /** Id */
            id: string;
            /** Label */
            label: string | null;
            started_at: components["schemas"]["UtcDatetime"] | null;
            status: components["schemas"]["RunStatus"];
            /**
             * Tags
             * @default []
             */
            tags: string[];
            totals: components["schemas"]["RunTotals"];
            updated_at: components["schemas"]["UtcDatetime"];
            /**
             * Warnings
             * @default []
             */
            warnings: string[];
        };
        /**
         * RunConfig
         * @description The effective configuration of a run, persisted verbatim.
         *
         *     Contains no secrets: :class:`~llm_eval_lab.models.common.ProviderConfig`
         *     records the NAME of the credential environment variable, never its value.
         */
        RunConfig: {
            case_selection: components["schemas"]["CaseSelection"];
            /**
             * Case Timeout S
             * @default 120
             */
            case_timeout_s: number;
            /**
             * Concurrency
             * @default 8
             */
            concurrency: number;
            /**
             * Error Policy
             * @default exclude
             * @enum {string}
             */
            error_policy: "exclude" | "fail";
            /** Evaluators */
            evaluators: components["schemas"]["EvaluatorSpec"][];
            /**
             * Judge Concurrency
             * @default 4
             */
            judge_concurrency: number;
            /** Library Version */
            library_version: string;
            /**
             * Max Cost
             * @description Running spend cap in the price table's currency; the runner cancels the run with error 'budget_exceeded' when accumulated known cost exceeds it. None disables the cap.
             */
            max_cost?: string | null;
            /**
             * Max Error Rate
             * @default 1
             */
            max_error_rate: number;
            /** Notes */
            notes?: string | null;
            params: components["schemas"]["GenerationParams"];
            /**
             * Plugins
             * @default []
             */
            plugins: components["schemas"]["PluginRecord"][];
            /** Price Table Hash */
            price_table_hash: string;
            /** Price Table Id */
            price_table_id: string;
            /** Price Table Version */
            price_table_version: string;
            provider: components["schemas"]["ProviderConfig-Output"];
            /** Python Version */
            python_version: string;
            /**
             * @default {
             *       "base_delay_s": 0.5,
             *       "jitter": "full",
             *       "max_attempts": 4,
             *       "max_delay_s": 30,
             *       "max_total_delay_s": 120,
             *       "respect_retry_after": true
             *     }
             */
            retry: components["schemas"]["RetryPolicy"];
            /**
             * Schema Version
             * @default 1
             * @constant
             */
            schema_version: 1;
            /** Seed */
            seed?: number | null;
            /** Suite Hash */
            suite_hash: string;
            /** Suite Name */
            suite_name: string;
            /** Suite Source */
            suite_source: string | null;
            /** Suite Version */
            suite_version: string;
        };
        /**
         * RunDetail
         * @description What ``GET /api/runs/{id}`` returns: the run plus its derived figures.
         *
         *     ``pass_rate`` is computed from the persisted case results under the run's own
         *     ``error_policy``, and ``n_pass_denominator`` says what it was computed over,
         *     so a rate is never shown without the population behind it.
         *
         *     ``run.config.suite_source`` is always ``None`` on this response. It is the
         *     server's own path to the suite file, which the database keeps and the API
         *     withholds; ``suite_name``, ``suite_version`` and ``suite_hash`` beside it say
         *     which suite ran.
         */
        RunDetail: {
            /** N Cases */
            n_cases: number;
            /** N Completed */
            n_completed: number;
            /** N Errors */
            n_errors: number;
            /** N Pass Denominator */
            n_pass_denominator: number;
            /** N Passed */
            n_passed: number;
            /** Pass Rate */
            pass_rate: number | null;
            run: components["schemas"]["Run"];
        };
        /**
         * RunLinks
         * @description Where to go next after launching a run, so a client hard-codes no paths.
         */
        RunLinks: {
            /** Cancel */
            cancel: string;
            /** Cases */
            cases: string;
            /** Export */
            export: string;
            /** Metrics */
            metrics: string;
            /** Self */
            self: string;
            /** Status */
            status: string;
        };
        /**
         * RunStatus
         * @description Lifecycle state of a benchmark run.
         * @enum {string}
         */
        RunStatus: "pending" | "running" | "completed" | "partial" | "failed" | "cancelled" | "interrupted";
        /**
         * RunStatusResponse
         * @description What ``GET /api/runs/{id}/status`` reports, for a polling client.
         *
         *     Read from the persisted run row, so it is equally correct for a run this
         *     process launched, a run the CLI launched and a run a previous process left
         *     behind. ``managed`` says whether this process holds the task, which is what
         *     decides whether cancelling can work.
         */
        RunStatusResponse: {
            /** Cancelled */
            cancelled: number;
            /** Completed */
            completed: number;
            /** Error */
            error: string | null;
            /** Errors */
            errors: number;
            /**
             * Last Case Id
             * @description The case named by the most recent progress event, for a run this process is executing. A display aid, not a count: with concurrency above one several cases are in flight and this names the one that most recently changed state. Null whenever managed is false.
             */
            last_case_id?: string | null;
            /** Managed */
            managed: boolean;
            /** Passed */
            passed: number;
            /** Run Id */
            run_id: string;
            /** Started At */
            started_at: string | null;
            status: components["schemas"]["RunStatus"];
            /** Total */
            total: number;
            /**
             * Updated At
             * Format: date-time
             */
            updated_at: string;
        };
        /**
         * RunSummary
         * @description The listing-sized projection of a run, denormalized for the runs table.
         */
        RunSummary: {
            completed_at: components["schemas"]["UtcDatetime"] | null;
            created_at: components["schemas"]["UtcDatetime"];
            /** Id */
            id: string;
            /** Label */
            label: string | null;
            /** Model */
            model: string;
            /** N Cases */
            n_cases: number;
            /** N Completed */
            n_completed: number;
            /** N Errors */
            n_errors: number;
            /** P95 Ms */
            p95_ms: number | null;
            /** Pass Rate */
            pass_rate: number | null;
            /** Provider */
            provider: string;
            status: components["schemas"]["RunStatus"];
            /** Suite Hash */
            suite_hash: string;
            /** Suite Name */
            suite_name: string;
            /** Suite Version */
            suite_version: string;
            /** Total Cost */
            total_cost: string | null;
        };
        /**
         * RunTotals
         * @description Running counters maintained as cases complete.
         */
        RunTotals: {
            /**
             * N Cancelled
             * @default 0
             */
            n_cancelled: number;
            /**
             * N Cases
             * @default 0
             */
            n_cases: number;
            /**
             * N Completed
             * @default 0
             */
            n_completed: number;
            /**
             * N Errors
             * @default 0
             */
            n_errors: number;
            /**
             * N Passed
             * @default 0
             */
            n_passed: number;
        };
        ShortText: string;
        SuiteName: string;
        /**
         * ThresholdDirection
         * @description Which way a metric has to move for the change to be an improvement.
         * @enum {string}
         */
        ThresholdDirection: "higher_is_better" | "lower_is_better";
        /**
         * TokenTotals
         * @description Token totals, with explicit counts of how many cases reported usage.
         */
        TokenTotals: {
            /** Input Tokens */
            input_tokens: number | null;
            /** Mean Input Tokens */
            mean_input_tokens: number | null;
            /** Mean Output Tokens */
            mean_output_tokens: number | null;
            /** N Missing Usage */
            n_missing_usage: number;
            /** N With Usage */
            n_with_usage: number;
            /** Output Tokens */
            output_tokens: number | null;
            /** Total Tokens */
            total_tokens: number | null;
        };
        /**
         * TokenUsage
         * @description Token counts as reported by the vendor.
         *
         *     ``total_tokens`` is derived when both halves are known and the vendor did
         *     not supply it. The reverse is never done: a lone total is never split into
         *     a fabricated input/output pair.
         */
        TokenUsage: {
            /** Cached Input Tokens */
            cached_input_tokens?: number | null;
            /** Input Tokens */
            input_tokens?: number | null;
            /** Output Tokens */
            output_tokens?: number | null;
            /** Reasoning Tokens */
            reasoning_tokens?: number | null;
            /** Total Tokens */
            total_tokens?: number | null;
        };
        /** Format: date-time */
        UtcDatetime: string;
        /**
         * ValidateRequest
         * @description Body of ``POST /api/benchmarks/validate``.
         *
         *     A NAME, never a path. The API resolves names under the configured benchmark
         *     root and accepts nothing that escapes it.
         */
        ValidateRequest: {
            name: components["schemas"]["SuiteName"];
        };
        /**
         * ValidateResponse
         * @description What ``POST /api/benchmarks/validate`` returns.
         *
         *     A ``200`` with ``valid: false`` rather than a ``422``: the client ASKED
         *     whether the file is valid, and "no" is a successful answer to that question.
         *     The ``422`` problem detail is what a client gets when it tries to RUN an
         *     invalid suite.
         */
        ValidateResponse: {
            /** Errors */
            errors: components["schemas"]["ApiFieldError"][];
            /** Name */
            name: string;
            /** Valid */
            valid: boolean;
        };
        /**
         * Verdict
         * @description Overall outcome of a regression comparison, and the CI exit-code source.
         * @enum {string}
         */
        Verdict: "pass" | "warn" | "fail" | "incomparable";
        /**
         * VersionResponse
         * @description What ``GET /api/version`` reports.
         */
        VersionResponse: {
            /** Python */
            python: string;
            /** Schema Version */
            schema_version: number;
            /** Version */
            version: string;
        };
    };
    responses: never;
    parameters: never;
    requestBodies: never;
    headers: never;
    pathItems: never;
}
export type $defs = Record<string, never>;
export interface operations {
    benchmarks_api_benchmarks_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Page_BenchmarkEntry_"];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    validate_api_benchmarks_validate_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["ValidateRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ValidateResponse"];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    benchmark_api_benchmarks__name__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                name: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["BenchmarkDetail"];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    compare_by_query_api_compare_get: {
        parameters: {
            query: {
                baseline_run_id: string;
                candidate_run_id: string;
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RegressionReport"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request conflicts with the current state of the run. */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    compare_api_compare_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["CompareRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RegressionReport"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request conflicts with the current state of the run. */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    evaluators_api_evaluators_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["EvaluatorEntry"][];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    health_api_health_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HealthResponse"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. The body is still the health report, not a problem detail: the caller asked for the health of the process and that question was answered. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["HealthResponse"];
                };
            };
        };
    };
    model_summary_api_models_summary_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ModelSummaryRow"][];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    pricing_api_pricing_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["PriceTable"];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    providers_api_providers_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProviderEntry"][];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    list_runs_api_runs_get: {
        parameters: {
            query?: {
                status?: components["schemas"]["RunStatus"] | null;
                provider?: string | null;
                model?: string | null;
                suite?: string | null;
                limit?: number;
                offset?: number;
                order?: "created_at" | "-created_at";
            };
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Page_RunSummary_"];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request conflicts with the current state of the run. */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A provider is unavailable, or its credential variable is unset. */
            424: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description Too many runs are already executing in this process. */
            429: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    create_run_api_runs_post: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody: {
            content: {
                "application/json": components["schemas"]["CreateRunRequest"];
            };
        };
        responses: {
            /** @description Successful Response */
            202: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CreateRunResponse"];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request conflicts with the current state of the run. */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A provider is unavailable, or its credential variable is unset. */
            424: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description Too many runs are already executing in this process. */
            429: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    get_run_api_runs__id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RunDetail"];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request conflicts with the current state of the run. */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A provider is unavailable, or its credential variable is unset. */
            424: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description Too many runs are already executing in this process. */
            429: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    cancel_run_api_runs__id__cancel_post: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            202: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CancelResponse"];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request conflicts with the current state of the run. */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A provider is unavailable, or its credential variable is unset. */
            424: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description Too many runs are already executing in this process. */
            429: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    list_cases_api_runs__id__cases_get: {
        parameters: {
            query?: {
                status?: components["schemas"]["CaseStatus"] | null;
                passed?: boolean | null;
                tag?: string | null;
                category?: string | null;
                q?: string | null;
                limit?: number;
                offset?: number;
            };
            header?: never;
            path: {
                id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["Page_CaseResultSummary_"];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request conflicts with the current state of the run. */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A provider is unavailable, or its credential variable is unset. */
            424: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description Too many runs are already executing in this process. */
            429: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    get_case_api_runs__id__cases__case_id__get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                id: string;
                case_id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["CaseResult"];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request conflicts with the current state of the run. */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A provider is unavailable, or its credential variable is unset. */
            424: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description Too many runs are already executing in this process. */
            429: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    export_run_api_runs__id__export_get: {
        parameters: {
            query?: {
                format?: "json" | "csv";
            };
            header?: never;
            path: {
                id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": unknown;
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request conflicts with the current state of the run. */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A provider is unavailable, or its credential variable is unset. */
            424: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description Too many runs are already executing in this process. */
            429: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    run_metrics_api_runs__id__metrics_get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["AggregateMetrics"];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request conflicts with the current state of the run. */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A provider is unavailable, or its credential variable is unset. */
            424: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description Too many runs are already executing in this process. */
            429: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    run_status_api_runs__id__status_get: {
        parameters: {
            query?: never;
            header?: never;
            path: {
                id: string;
            };
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["RunStatusResponse"];
                };
            };
            /** @description The request was malformed, or a suite name was not usable. */
            400: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description No run, case or benchmark suite has that identifier. */
            404: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request conflicts with the current state of the run. */
            409: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A request body was sent without declaring its length. */
            411: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request body exceeds the configured cap. */
            413: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The request or the benchmark suite it names did not validate. */
            422: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description A provider is unavailable, or its credential variable is unset. */
            424: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description Too many runs are already executing in this process. */
            429: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description The database could not be reached. */
            503: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
    version_api_version_get: {
        parameters: {
            query?: never;
            header?: never;
            path?: never;
            cookie?: never;
        };
        requestBody?: never;
        responses: {
            /** @description Successful Response */
            200: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["VersionResponse"];
                };
            };
            /** @description A bearer token is required and was absent or wrong. */
            401: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
            /** @description An unmapped failure, reported with a correlation id and nothing else. */
            500: {
                headers: {
                    [name: string]: unknown;
                };
                content: {
                    "application/json": components["schemas"]["ProblemDetail"];
                };
            };
        };
    };
}
